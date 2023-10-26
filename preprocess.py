import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from easydict import EasyDict as edict

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


def get_rgba(image, alpha_matting=False):
    try:
        from rembg import remove
    except ImportError:
        print('Please install rembg with "pip install rembg"')
        sys.exit()
    return remove(image, alpha_matting=alpha_matting)


from midas.model_loader import default_models, load_model

depth_config={
    "input_path": None,
    "output_path": None,
    "model_weights": "load/midas/dpt_beit_large_512.pt",
    "model_type": "dpt_beit_large_512",
    "side": False,
    "optimize": False,
    "height": None,
    "square": False,
    "device":0,
    "grayscale": False
}

class DepthEstimator:
    def __init__(self,**kwargs):
        # update coming args
        for key, value in kwargs.items():
            depth_config[key]=value
            
        # self.config=DefaultMunch.fromDict(depth_config)
        self.config = edict(depth_config) 
        
        # select device
        self.device = torch.device(self.config.device)
        model, transform, net_w, net_h = load_model(f"cuda:{self.config.device}", self.config.model_weights, self.config.model_type, 
                                                    self.config.optimize, self.config.height, self.config.square)
        self.model, self.transform, self.net_w, self.net_h=model, transform, net_w, net_h
        self.first_execution = True
        
    @torch.no_grad()
    def process(self,image,target_size):
        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)


        if self.first_execution:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            self.first_execution = False

        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction
    
    @torch.no_grad()
    def get_monocular_depth(self,rgb, output_path=None):
        original_image_rgb=rgb
        image = self.transform({"image": original_image_rgb})["image"]
        
        prediction = self.process(image, original_image_rgb.shape[1::-1])
        return prediction
    
def process_single_image(image_path, depth_estimator):
    out_dir = os.path.dirname(os.path.dirname(image_path))
    image_name = image_path.split('/')[-1]

    depth_path = os.path.join(out_dir, 'depth')
    os.makedirs(depth_path, exist_ok=True)
    depth_img_path = os.path.join(depth_path, image_name)

    print(f'[INFO] loading image {image_path}...')
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f'[INFO] background removal...')
        rgba = BackgroundRemoval()(image)  # [H, W, 4]

    # Predict depth using Midas
    mask = rgba[..., -1] > 0
    depth = depth_estimator.get_monocular_depth(image / 255)
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)

    cv2.imwrite(depth_img_path, depth)

def gen_dtu_rgba(path="load/DTU/dtu_scan105"):
    image_path = os.path.join(path, 'image')
    image_list = sorted(os.listdir(image_path))

    mask_path = os.path.join(path, 'mask')
    mask_list = sorted(os.listdir(mask_path))

    rgba_path = os.path.join(path, 'rgba')
    os.makedirs(rgba_path, exist_ok=True)

    for i in range(len(image_list)):
        rgb = cv2.imread(os.path.join(image_path, image_list[i]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(mask_path, mask_list[i]), cv2.IMREAD_UNCHANGED)[...,0]

        mask[mask < 127] = 0
        rgb[mask < 127] = 255

        rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask

        rgba = cv2.resize(rgba, (800, 600))
        cv2.imwrite(os.path.join(rgba_path, image_list[i]), rgba)

if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str, nargs='*', help="path to image (png, jpeg, etc.)")
    parser.add_argument('--folder', default="load/DTU/dtu_scan122/rgba", type=str, help="path to a folder of image (png, jpeg, etc.)")
    parser.add_argument('--imagepattern', default="*.png", type=str, help="image name pattern")
    parser.add_argument('--exclude', default='', type=str, nargs='*', help="path to image (png, jpeg, etc.) to exclude")
    opt = parser.parse_args()

    gen_dtu_rgba(path="load/DTU/dtu_scan122")

    depth_estimator = DepthEstimator()
    
    if opt.path is not None:
        paths = opt.path
    else:
        paths = glob.glob(os.path.join(opt.folder, f'{opt.imagepattern}')) 
        for exclude_path in opt.exclude:
            if exclude_path in paths:
                del paths[exclude_path]
    for path in sorted(paths):
        process_single_image(path, depth_estimator)