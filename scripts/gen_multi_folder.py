#! /bin/python3
import os
import argparse
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'GNU' 

def R_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    - R: A 3x3 numpy array representing a rotation matrix.

    Returns:
    - A numpy array representing the quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])

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

    print("合并所有sparse......")
    all_file_path = os.path.join(args.data_root_dir, "folder_all")
    os.makedirs(all_file_path, exist_ok=True)
    os.system(f"rm -rf {os.path.join(all_file_path, 'images')}")
    os.system(f"rm -rf {os.path.join(all_file_path, 'sparse')}")
    os.system(f"cp -r {os.path.join(args.data_root_dir, 'folder_0/sparse')} {all_file_path}")
    os.system(f"cp -r {os.path.join(args.data_root_dir, 'folder_0/images')} {all_file_path}")
    for folder_num in tqdm(range(1, len(images))):
        single_file_path = os.path.join(args.data_root_dir, f"folder_{folder_num}")

        # deal with images
        images_1 = os.listdir(os.path.join(all_file_path, "images"))
        images_1.sort(key=lambda x: int(x.split(".")[0]))
        images_2 = os.listdir(os.path.join(single_file_path, "images"))
        images_2.sort(key=lambda x: int(x.split(".")[0]))
        start_number = int(images_1[-1].split(".")[0])
        for image in images_2:
            Image.open(os.path.join(single_file_path, "images", image)).save(os.path.join(all_file_path, "images", f"{start_number}.png"))
            start_number += 1

        # deal with cameras.txt
        '''
        CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        '''
        with open(os.path.join(all_file_path, "sparse/0/cameras.txt"), "r") as f:
            lines = f.readlines()
            others = lines[0].split(" ")[1:]
            start_number = int(lines[-1].split(" ")[0])
        with open(os.path.join(all_file_path, "sparse/0/cameras.txt"), "a") as f:
            for i in range(len(images[folder_num])):
                f.write(f"{start_number} {others[0]} {others[1]} {others[2]} {others[3]} {others[4]} {others[5]} {others[6]}")
                start_number += 1

        # deal with images.txt
        '''
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] as (X, Y, POINT3D_ID)  EMPTY
        '''
        with open(os.path.join(all_file_path, "sparse/0/images.txt"), "r") as f:
            lines_ori = f.readlines()
            lines = [i for i in lines_ori if i != "\n"]
            lines.sort(key=lambda x: int(x.split(" ")[-1].split(".")[0]))
        with open(os.path.join(single_file_path, "sparse/0/images.txt"), "r") as f:
            lines_ori = f.readlines()
            lines_single = [i for i in lines_ori if i != "\n"]
            lines_single.sort(key=lambda x: int(x.split(" ")[-1].split(".")[0]))
        
        with open(os.path.join(all_file_path, "sparse/0/images.txt"), "a") as f:
            num = int(lines[-1].split(" ")[0])
            qw, qx, qy, qz, tx, ty, tz = [float(i) for i in lines[-1].split(" ")[1:8]]
            RT_ori = np.array([ [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
                                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, ty],
                                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, tz],
                                [0, 0, 0, 1]])
            
            RT_ori = np.linalg.inv(RT_ori)
            RT = dict()
            for line in lines_single:
                qw, qx, qy, qz, tx, ty, tz = [float(i) for i in line.split(" ")[1:8]]
                RT_single = np.array([ [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
                                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, ty],
                                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, tz],
                                        [0, 0, 0, 1]])
                RT_single = np.linalg.inv(RT_single)
                RT[line.split(" ")[-1].split(".")[0]] = RT_single
            
            C = np.dot(RT_ori, np.linalg.inv(RT[lines[-1].split(" ")[-1].split(".")[0]]))

            for i in range(len(lines_single)):
                RT_new = np.dot(C, RT[lines_single[i].split(" ")[-1].split(".")[0]])
                RT_new = np.linalg.inv(RT_new)
                qw, qx, qy, qz = R_to_quaternion(RT_new[:3, :3])
                f.write(f'{num} {qw} {qx} {qy} {qz} {RT_new[0, 3]} {RT_new[1, 3]} {RT_new[2, 3]} {num} {lines_single[i].split(" ")[-1]}\n')
                num += 1
        
        # deal with points3D.ply

        ply_ori = PlyData.read(os.path.join(all_file_path, "sparse/0/points3D.ply"))
        ply_single = PlyData.read(os.path.join(single_file_path, "sparse/0/points3D.ply"))
        ply_target = os.path.join(all_file_path, "sparse/0/points3D.ply")

        data_x_ori = ply_ori.elements[0]['x']
        data_x_ori = data_x_ori.reshape(-1, 1)
        data_y_ori = ply_ori.elements[0]['y']
        data_y_ori = data_y_ori.reshape(-1, 1)
        data_z_ori = ply_ori.elements[0]['z']
        data_z_ori = data_z_ori.reshape(-1, 1)

        ones = np.ones(data_x_ori.shape[0]).reshape(-1, 1)
        points_ori = np.concatenate((data_x_ori, data_y_ori, data_z_ori, ones), axis=1)

        data_x_single = ply_single.elements[0]['x']
        data_x_single = data_x_single.reshape(-1, 1)
        data_y_single = ply_single.elements[0]['y']
        data_y_single = data_y_single.reshape(-1, 1)
        data_z_single = ply_single.elements[0]['z']
        data_z_single = data_z_single.reshape(-1, 1)

        ones = np.ones(data_x_single.shape[0]).reshape(-1, 1)
        points_single = np.concatenate((data_x_single, data_y_single, data_z_single, ones), axis=1)

        step = 20
        points_single = np.dot(C, points_single.T).T[::step]

        if folder_num == 1:
            points_ori = points_ori[::step]
        points_all = np.concatenate((points_ori, points_single), axis=0)
        data_x = points_all[:, 0].reshape(-1, 1)
        data_y = points_all[:, 1].reshape(-1, 1)
        data_z = points_all[:, 2].reshape(-1, 1)

        if folder_num == 1:
            data_nx = np.concatenate((ply_ori.elements[0]['nx'][::step], ply_single.elements[0]['nx'][::step]), axis=0)
            data_nx = data_nx.reshape(-1, 1)
            data_ny = np.concatenate((ply_ori.elements[0]['ny'][::step], ply_single.elements[0]['ny'][::step]), axis=0)
            data_ny = data_ny.reshape(-1, 1)
            data_nz = np.concatenate((ply_ori.elements[0]['nz'][::step], ply_single.elements[0]['nz'][::step]), axis=0)
            data_nz = data_nz.reshape(-1, 1)
            data_red = np.concatenate((ply_ori.elements[0]['red'][::step], ply_single.elements[0]['red'][::step]), axis=0)
            data_red = data_red.reshape(-1, 1)
            data_green = np.concatenate((ply_ori.elements[0]['green'][::step], ply_single.elements[0]['green'][::step]), axis=0)
            data_green = data_green.reshape(-1, 1)
            data_blue = np.concatenate((ply_ori.elements[0]['blue'][::step], ply_single.elements[0]['blue'][::step]), axis=0)
            data_blue = data_blue.reshape(-1, 1)
            
        else:
            data_nx = np.concatenate((ply_ori.elements[0]['nx'], ply_single.elements[0]['nx'][::step]), axis=0)
            data_nx = data_nx.reshape(-1, 1)
            data_ny = np.concatenate((ply_ori.elements[0]['ny'], ply_single.elements[0]['ny'][::step]), axis=0)
            data_ny = data_ny.reshape(-1, 1)
            data_nz = np.concatenate((ply_ori.elements[0]['nz'], ply_single.elements[0]['nz'][::step]), axis=0)
            data_nz = data_nz.reshape(-1, 1)
            data_red = np.concatenate((ply_ori.elements[0]['red'], ply_single.elements[0]['red'][::step]), axis=0)
            data_red = data_red.reshape(-1, 1)
            data_green = np.concatenate((ply_ori.elements[0]['green'], ply_single.elements[0]['green'][::step]), axis=0)
            data_green = data_green.reshape(-1, 1)
            data_blue = np.concatenate((ply_ori.elements[0]['blue'], ply_single.elements[0]['blue'][::step]), axis=0)
            data_blue = data_blue.reshape(-1, 1)

        data = np.concatenate((data_x, data_y, data_z, data_nx, data_ny, data_nz, data_red, data_green, data_blue), axis=1)
        # data = np.concatenate((data_x[::step], data_y[::step], data_z[::step], data_nx[::step], data_ny[::step], data_nz[::step], data_red[::step], data_green[::step], data_blue[::step]), axis=1)
        # print(data.shape)
        elements = np.empty(data_x.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        elements[:] = list(map(tuple, data))
        el = PlyElement.describe(elements, name='vertex')
        PlyData([el]).write(ply_target)