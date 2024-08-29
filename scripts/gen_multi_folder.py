#! /bin/python3
import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'GNU' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default="./", help="path to the images")
    parser.add_argument("--images_num", type=int, default=2, help="number of images for each folder")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    images_all = os.listdir(os.path.join(args.data_root_dir, "images"))
    images_all.sort(key=lambda x: int(x.split(".")[0]))
    print("images num: ", len(images_all))
        
    print("分割数据集ing......")
    # 分成多个组，每组args.images_num张图片
    num = 0
    images = []
    while num < len(images_all):
        if num == 0:
            images.append(images_all[num:num+args.images_num])
        else:
            num -= 1
            images.append(images_all[num:num+args.images_num])
        num += args.images_num
    for folder_num in tqdm(range(len(images))):
        folder_path = os.path.join(args.data_root_dir, f"folder_{folder_num}/images")
        os.makedirs(folder_path, exist_ok=True)
        for image in images[folder_num]:
            Image.open(os.path.join(args.data_root_dir, "images", image)).save(os.path.join(folder_path, image))
    
    print("运行DUSt3R......")
    for folder_num in tqdm(range(len(images))):
        os.system(f'CUDA_VISIBLE_DEVICES={args.gpu_id} python -W ignore ./coarse_init_infer.py \
            --img_base_path {os.path.join(args.data_root_dir, "folder_"+str(folder_num))} \
            --n_views {len(images[folder_num])}  \
            --focal_avg \
            --focal 351.6702 \
            ')