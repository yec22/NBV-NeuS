import os
import cv2
import argparse
import numpy as np
from shutil import copyfile

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

def center_crop(path):
    image_path = os.path.join(path, 'image')
    image_list = sorted(os.listdir(image_path))

    mask_path = os.path.join(path, 'mask')
    mask_list = sorted(os.listdir(mask_path))

    crop_path = os.path.join(path, 'image_crop')
    os.makedirs(crop_path, exist_ok=True)

    crop_mask_path = os.path.join(path, 'mask_crop')
    os.makedirs(crop_mask_path, exist_ok=True)

    for i in range(len(image_list)):
        rgb = cv2.imread(os.path.join(image_path, image_list[i]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(mask_path, mask_list[i]), cv2.IMREAD_UNCHANGED)[...,0]

        resize_rgb = cv2.resize(rgb, (800, 600))
        resize_mask = cv2.resize(mask, (800, 600))

        W, H = 800, 600
        
        crop_rgb = resize_rgb[:, W//2-H//2:W//2+H//2, :]
        crop_resize_rgb = cv2.resize(crop_rgb, (384, 384))
        cv2.imwrite(os.path.join(crop_path, image_list[i]), crop_resize_rgb)

        crop_mask = resize_mask[:, W//2-H//2:W//2+H//2]
        crop_resize_mask = cv2.resize(crop_mask, (384, 384))
        cv2.imwrite(os.path.join(crop_mask_path, image_list[i]), crop_resize_mask)
        

def process_normal(path):
    image_path = os.path.join(path, 'normal_png')
    image_list = sorted(os.listdir(image_path))

    mask_path = os.path.join(path, 'mask_crop')
    mask_list = sorted(os.listdir(mask_path))

    normal_path = os.path.join(path, 'normal')
    os.makedirs(normal_path, exist_ok=True)

    for i in range(len(image_list)):
        norm_data = (cv2.imread(os.path.join(image_path, image_list[i])) / 255.0 - 0.5) * 2
        norm_data = -norm_data
        mask = cv2.imread(os.path.join(mask_path, mask_list[i]), cv2.IMREAD_UNCHANGED)
        mask[mask < 127] = 0
        norm_data[mask < 127] = 0
        np.savez(f"{normal_path}/{image_list[i][:6]}.npz", arr_0=norm_data[...,::-1])
        norm_img = (norm_data / 2 + 0.5) * 255
        cv2.imwrite(os.path.join(normal_path, f'{image_list[i][:6]}.png'), norm_img.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str, nargs='*', help="path to image (png, jpeg, etc.)")
    parser.add_argument('--folder', default="load/DTU/dtu_scan122/rgba", type=str, help="path to a folder of image (png, jpeg, etc.)")
    parser.add_argument('--imagepattern', default="*.png", type=str, help="image name pattern")
    parser.add_argument('--exclude', default='', type=str, nargs='*', help="path to image (png, jpeg, etc.) to exclude")
    opt = parser.parse_args()

    # gen_dtu_rgba(path="load/DTU/dtu_scan105")
    # center_crop(path="load/DTU/dtu_scan105")
    process_normal(path="load/DTU/dtu_scan105")