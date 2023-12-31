import os
import json, copy
import math
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

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

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
    eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
    rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
    up = rot_axis
    rot_dir = torch.cross(rot_axis, cam_center)
    max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

    all_c2w = []
    for theta in torch.linspace(-max_angle, max_angle, n_steps):
        cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class DTUDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        cams = np.load(os.path.join(self.config.root_dir, self.config.cameras_file))

        img_sample = cv2.imread(os.path.join(self.config.root_dir, 'image', '000000.png'))
        H, W = img_sample.shape[0], img_sample.shape[1]

        if self.split == 'train':
            w, h = 384, 384
        else:
            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")
        
        if self.split == 'val':
            w, h = w//4, h//4

        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.factor = h / H

        if self.split == 'train':
            mask_dir = os.path.join(self.config.root_dir, 'mask_crop')
        else:
            mask_dir = os.path.join(self.config.root_dir, 'mask')
        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        depth_dir = os.path.join(self.config.root_dir, 'depth')
        normal_dir = os.path.join(self.config.root_dir, 'normal')
        
        self.directions = []
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        self.pred_normals = []
        self.intrinsics = []

        n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

        self.candidate_views = [i for i in range(n_images)]

        for i in range(n_images):
            world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            if self.split == "train":
                fx, fy = K[0,0] * self.factor, K[1,1] * self.factor
                cx = (K[0, 2] - (W - H) // 2) * self.factor
                cy = K[1, 2] * self.factor
            else:
                fx, fy = K[0,0] * self.factor, K[1,1] * self.factor
                cx = K[0, 2] * self.factor
                cy = K[1, 2] * self.factor
            self.intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ]
                    ,dtype=torch.float32
                )
            )
            directions = get_ray_directions(w, h, fx, fy, cx, cy)
            self.directions.append(directions)
            
            c2w = torch.from_numpy(c2w).float()

            # blender follows opengl camera coordinates (right up back)
            # NeuS DTU data coordinate system (right down front) is different from blender
            # https://github.com/Totoro97/NeuS/issues/9
            # for c2w, flip the sign of input camera coordinate yz
            c2w_ = c2w.clone()
            c2w_[:3,1:3] *= -1. # flip input sign
            self.all_c2w.append(c2w_[:3,:4])       

            if self.split == 'train':
                img_path = os.path.join(self.config.root_dir, 'image_crop', f'{i:06d}.png')
            else:
                img_path = os.path.join(self.config.root_dir, 'image', f'{i:06d}.png')
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]

            if self.split == 'train':
                mask_path = os.path.join(mask_dir, f'{i:06d}.png')
            else:
                mask_path = os.path.join(mask_dir, f'{i:03d}.png')
            mask = Image.open(mask_path).convert('L') # (H, W, 1)
            mask = mask.resize(self.img_wh, Image.BICUBIC)
            mask = TF.to_tensor(mask)[0]
            
            if self.split == 'train':
                pred_normal_path = os.path.join(normal_dir, f'{i:06d}.npz')
                normal = np.load(pred_normal_path)['arr_0']
                normal[:, :, 1:3] *= -1.
                ex_i = torch.linalg.inv(c2w_)
                normal_world = get_world_normal(normal.reshape(-1, 3), ex_i).reshape(self.img_wh[1], self.img_wh[0], 3)
            else:
                normal_world = np.zeros((self.img_wh[1], self.img_wh[0], 3))
            normal_world = torch.from_numpy(normal_world).float()

            self.all_fg_masks.append(mask) # (h, w)
            self.all_images.append(img)
            self.pred_normals.append(normal_world)

        self.all_c2w_all = torch.stack(self.all_c2w, dim=0)
        self.all_images_all = torch.stack(self.all_images, dim=0)
        self.all_fg_masks_all = torch.stack(self.all_fg_masks, dim=0)  
        self.pred_normals_all = torch.stack(self.pred_normals, dim=0)
        self.directions_all = torch.stack(self.directions, dim=0)
        self.intrinsics = torch.stack(self.intrinsics, dim=0)

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
            n = self.config.get('n_view', 3)
            scene_id = self.config.get('scene')

            if isinstance(scene_id, int) and scene_id > 80:
                camera_pos_front = self.all_c2w_all[:49, :, 3]
                camera_pos_back = self.all_c2w_all[49:, :, 3]

                kmeans_front = KMeans(n_clusters=n//2, random_state=0, n_init="auto").fit(camera_pos_front)
                kmeans_back = KMeans(n_clusters=n-n//2, random_state=0, n_init="auto").fit(camera_pos_back)

                center_front = kmeans_front.cluster_centers_
                center_back = kmeans_back.cluster_centers_

                initial_view = []

                for i in range(n//2):
                    idx = torch.argmin(torch.sum((camera_pos_front - center_front[i]) ** 2, dim=-1))
                    initial_view.append(idx.item())
                
                for i in range(n-n//2):
                    idx = torch.argmin(torch.sum((camera_pos_back - center_back[i]) ** 2, dim=-1))
                    initial_view.append(49+idx.item())
            
            else:
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
                self.pred_normals_train = self.pred_normals_all[self.initial_view,...]
                self.all_fg_masks_train = self.all_fg_masks_all[self.initial_view,...]
                self.directions_train = self.directions_all[self.initial_view,...]
            else:
                self.all_c2w_train = self.all_c2w_all
                self.all_images_train = self.all_images_all
                self.pred_normals_train = self.pred_normals_all
                self.all_fg_masks_train = self.all_fg_masks_all
                self.directions_train = self.directions_all
        
        # only validate with candidate views
        elif self.split == 'val':
            self.all_c2w = self.all_c2w_all[self.candidate_views,...]
            self.all_images = self.all_images_all[self.candidate_views,...]
            self.pred_normals = self.pred_normals_all[self.candidate_views,...]
            self.all_fg_masks = self.all_fg_masks_all[self.candidate_views,...]
            self.directions = self.directions_all[self.candidate_views,...]
        
        else: # test
            self.all_c2w = self.all_c2w_all[-1:,...]
            self.all_images = self.all_images_all[-1:,...]
            self.pred_normals = self.pred_normals_all[-1:,...]
            self.all_fg_masks = self.all_fg_masks_all[-1:,...]
            self.directions = self.directions_all[-1:,...]
        
        if self.split == 'train' or self.split == 'train_high':
            self.directions_train = self.directions_train.float().to(self.rank)
            self.pred_normals_train = self.pred_normals_train.float().to(self.rank)
            self.all_c2w_train, self.all_images_train, self.all_fg_masks_train = \
                self.all_c2w_train.float().to(self.rank), \
                self.all_images_train.float().to(self.rank), \
                self.all_fg_masks_train.float().to(self.rank)
        else:
            self.directions = self.directions.float().to(self.rank)
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
                self.pred_normals_train = self.pred_normals_all[self.initial_view,...]
                self.all_fg_masks_train = self.all_fg_masks_all[self.initial_view,...]
                self.directions_train = self.directions_all[self.initial_view,...]
            
            if self.split == 'val':
                self.all_c2w = self.all_c2w_all[self.candidate_views,...]
                self.all_images = self.all_images_all[self.candidate_views,...]
                self.pred_normals = self.pred_normals_all[self.candidate_views,...]
                self.all_fg_masks = self.all_fg_masks_all[self.candidate_views,...]
                self.directions = self.directions_all[self.candidate_views,...]
            
            if self.split == 'train' or self.split == 'train_high':
                self.directions_train = self.directions_train.float().to(self.rank)
                self.pred_normals_train = self.pred_normals_train.float().to(self.rank)
                self.all_c2w_train, self.all_images_train, self.all_fg_masks_train = \
                    self.all_c2w_train.float().to(self.rank), \
                    self.all_images_train.float().to(self.rank), \
                    self.all_fg_masks_train.float().to(self.rank)
            else:
                self.directions = self.directions.float().to(self.rank)
                self.pred_normals = self.pred_normals.float().to(self.rank)
                self.all_c2w, self.all_images, self.all_fg_masks = \
                    self.all_c2w.float().to(self.rank), \
                    self.all_images.float().to(self.rank), \
                    self.all_fg_masks.float().to(self.rank)


class DTUDataset(Dataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class DTUIterableDataset(IterableDataset, DTUDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('dtu')
class DTUDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = DTUIterableDataset(self.config, 'train')
            self.train_high_dataset = DTUIterableDataset(self.config, 'train_high')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = DTUDataset(self.config, self.config.get('val_split', 'val'))
        if stage in [None, 'test']:
            self.test_dataset = DTUDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = DTUDataset(self.config, 'train')

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
