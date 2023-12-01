import os, cv2
import json
import math
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
import copy
import pytorch_lightning as pl
import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

def get_world_normal(normal, extrin):
    '''
    Args:
        normal: N*3
        extrinsics: 4*4, world to camera
    Return:
        normal: N*3, in world space 
    '''
    extrinsics = copy.deepcopy(extrin)
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.cpu().numpy()
        
    assert extrinsics.shape[0] ==4
    normal = normal.transpose()
    extrinsics[:3, 3] = np.zeros(3)  # only rotation, no translation

    normal_world = np.matmul(np.linalg.inv(extrinsics),
                            np.vstack((normal, np.ones((1, normal.shape[1])))))[:3]
    normal_world = normal_world.transpose((1, 0))

    return normal_world

class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        if self.split == 'val':
            rewrite_split = 'train'
        elif self.split == 'train_high':
            rewrite_split = 'train'
        else:
            rewrite_split = self.split
        with open(os.path.join(self.config.root_dir, f"transforms_{rewrite_split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if self.split == 'train':
            w, h = 384, 384
        else:
            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = W // self.config.img_downscale, H // self.config.img_downscale
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")
        
        if split == 'val':
            w, h = w//4, h//4

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.pred_depths = []
        self.pred_normals = []

        n_images = len(meta['frames'])
        self.candidate_views = [i for i in range(n_images)]

        self.intrinsics = []
        self.intrinsics.append(
            torch.tensor(
                [
                    [self.focal, 0, self.w//2],
                    [0, self.focal, self.h//2],
                    [0, 0, 1]
                ]
                ,dtype=torch.float32
            )
        )

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix']))
            self.all_c2w.append(c2w[:3, :4])

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            
            if self.split == 'train':
                img_path_post = os.path.join('image_crop', img_path.split('/')[-1])
                img_path = os.path.join(self.config.root_dir, img_path_post)

            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            depth = torch.zeros((self.img_wh[1], self.img_wh[0]), dtype=torch.float32)

            self.all_fg_masks.append(img[..., -1]) # (h, w) -> alpha
            self.all_images.append(img[...,:3]) # (h, w, 3) -> RGB
            self.pred_depths.append(depth)
            if self.split == 'train':
                img_path_post = os.path.join('normal', img_path.split('/')[-1][:-4]+'.npz')
                pred_normal_path = os.path.join(self.config.root_dir, img_path_post)
                normal = np.load(pred_normal_path)['arr_0']
                normal[:, :, 1:3] *= -1.
                ex_i = torch.linalg.inv(c2w)
                normal_world = get_world_normal(normal.reshape(-1, 3), ex_i).reshape(self.img_wh[1], self.img_wh[0], 3)
            else:
                normal_world = np.zeros((self.img_wh[1], self.img_wh[0], 3))
            normal_world = torch.from_numpy(normal_world).float()
            self.pred_normals.append(normal_world)

        self.all_c2w_all, self.all_images_all, self.all_fg_masks_all = \
            torch.stack(self.all_c2w, dim=0), \
            torch.stack(self.all_images, dim=0), \
            torch.stack(self.all_fg_masks, dim=0)
        self.intrinsics = torch.stack(self.intrinsics, dim=0)
        self.pred_depths_all = torch.stack(self.pred_depths, dim=0)
        self.pred_normals_all = torch.stack(self.pred_normals, dim=0)

        initial_view = self.config.get('initial_view', None)

        if initial_view == "farthest":
            camera_pos = self.all_c2w_all[:, :, 3]
            farthest_dist = 0.
            img_pair = None
            for i in range(camera_pos.shape[0]):
                for j in range(i+1, camera_pos.shape[0]):
                    dist = torch.sum((camera_pos[i] - camera_pos[j]) ** 2)
                    if dist > farthest_dist:
                        farthest_dist = dist
                        img_pair = [i, j]
            
            farthest_dist = 0.
            for i in range(camera_pos.shape[0]):
                if i in img_pair:
                    continue
                dist = torch.sqrt(torch.sum((camera_pos[i] - camera_pos[img_pair[0]]) ** 2)) + \
                       torch.sqrt(torch.sum((camera_pos[i] - camera_pos[img_pair[1]]) ** 2))
                if dist > farthest_dist:
                    farthest_dist = dist
                    initial_view = [img_pair[0], img_pair[1], i]
                    
            initial_view = sorted(initial_view)
        
        if initial_view == "cluster":
            n = self.config.get('n_view', 4)

            camera_pos = self.all_c2w_all[:, :, 3]
            kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(camera_pos)
            center = kmeans.cluster_centers_
            initial_view = []

            for i in range(n):
                idx = torch.argmin(torch.sum((camera_pos - center[i]) ** 2, dim=-1))
                initial_view.append(idx.item())

            initial_view = sorted(initial_view)

        if initial_view:
            self.initial_view = copy.deepcopy(initial_view)
            for elem in self.initial_view:
                self.candidate_views.remove(elem)
        else:
            self.initial_view = None

        print(self.split, self.initial_view)

        # only train with limited views
        if self.split == 'train' or self.split == 'train_high':
            if self.initial_view:
                self.all_c2w_train = self.all_c2w_all[self.initial_view,...]
                self.all_images_train = self.all_images_all[self.initial_view,...]
                self.pred_depths_train = self.pred_depths_all[self.initial_view,...]
                self.pred_normals_train = self.pred_normals_all[self.initial_view,...]
                self.all_fg_masks_train = self.all_fg_masks_all[self.initial_view,...]
            else:
                self.all_c2w_train = self.all_c2w_all
                self.all_images_train = self.all_images_all
                self.pred_depths_train = self.pred_depths_all
                self.pred_normals_train = self.pred_normals_all
                self.all_fg_masks_train = self.all_fg_masks_all
            self.directions_train = self.directions
        
        # only validate with candidate views
        elif self.split == 'val':
            self.all_c2w = self.all_c2w_all[self.candidate_views,...]
            self.all_images = self.all_images_all[self.candidate_views,...]
            self.pred_depths = self.pred_depths_all[self.candidate_views,...]
            self.pred_normals = self.pred_normals_all[self.candidate_views,...]
            self.all_fg_masks = self.all_fg_masks_all[self.candidate_views,...]
        
        else: # test
            self.all_c2w = self.all_c2w_all
            self.all_images = self.all_images_all
            self.pred_depths = self.pred_depths_all
            self.pred_normals = self.pred_normals_all
            self.all_fg_masks = self.all_fg_masks_all

        if self.split == 'train' or self.split == 'train_high':
            self.directions_train = self.directions_train.float().to(self.rank)
            self.pred_depths_train = self.pred_depths_train.float().to(self.rank)
            self.pred_normals_train = self.pred_normals_train.float().to(self.rank)
            self.all_c2w_train, self.all_images_train, self.all_fg_masks_train = \
                self.all_c2w_train.float().to(self.rank), \
                self.all_images_train.float().to(self.rank), \
                self.all_fg_masks_train.float().to(self.rank)
        else: # test / val
            self.directions = self.directions.float().to(self.rank)
            self.pred_depths = self.pred_depths.float().to(self.rank)
            self.pred_normals = self.pred_normals.float().to(self.rank)
            self.all_c2w, self.all_images, self.all_fg_masks = \
                self.all_c2w.float().to(self.rank), \
                self.all_images.float().to(self.rank), \
                self.all_fg_masks.float().to(self.rank)
    
    def update(self, initial_view, candidate_views):
        if self.initial_view and (self.initial_view != initial_view):
            self.initial_view = copy.deepcopy(initial_view)
            self.candidate_views = copy.deepcopy(candidate_views)

            if self.split == 'train' or self.split == 'train_high':
                self.all_c2w_train = self.all_c2w_all[self.initial_view,...]
                self.all_images_train = self.all_images_all[self.initial_view,...]
                self.pred_depths_train = self.pred_depths_all[self.initial_view,...]
                self.pred_normals_train = self.pred_normals_all[self.initial_view,...]
                self.all_fg_masks_train = self.all_fg_masks_all[self.initial_view,...]
            
            if self.split == 'val':
                self.all_c2w = self.all_c2w_all[self.candidate_views,...]
                self.all_images = self.all_images_all[self.candidate_views,...]
                self.pred_depths = self.pred_depths_all[self.candidate_views,...]
                self.pred_normals = self.pred_normals_all[self.candidate_views,...]
                self.all_fg_masks = self.all_fg_masks_all[self.candidate_views,...]
            
            if self.split == 'train' or self.split == 'train_high':
                self.directions_train = self.directions_train.float().to(self.rank)
                self.pred_depths_train = self.pred_depths_train.float().to(self.rank)
                self.pred_normals_train = self.pred_normals_train.float().to(self.rank)
                self.all_c2w_train, self.all_images_train, self.all_fg_masks_train = \
                    self.all_c2w_train.float().to(self.rank), \
                    self.all_images_train.float().to(self.rank), \
                    self.all_fg_masks_train.float().to(self.rank)
            
            if self.split == 'val':
                self.directions = self.directions.float().to(self.rank)
                self.pred_depths = self.pred_depths.float().to(self.rank)
                self.pred_normals = self.pred_normals.float().to(self.rank)
                self.all_c2w, self.all_images, self.all_fg_masks = \
                    self.all_c2w.float().to(self.rank), \
                    self.all_images.float().to(self.rank), \
                    self.all_fg_masks.float().to(self.rank)
        

class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
            self.train_high_dataset = BlenderIterableDataset(self.config, 'train_high')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)
    
    def train_high_dataloader(self):
        return self.general_loader(self.train_high_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
