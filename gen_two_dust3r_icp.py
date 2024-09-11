import os
import numpy as np
from PIL import Image
import open3d as o3d
from plyfile import PlyData, PlyElement

view_path = "/home/yugrp01/dataset"

view_1_path = os.path.join(view_path, "folder_1")
view_2_path = os.path.join(view_path, "folder_2")

view_1_images_path = os.path.join(view_1_path, "images")
view_2_images_path = os.path.join(view_2_path, "images")

view_1_file_path = os.path.join(view_1_path, "sparse/0")
view_2_file_path = os.path.join(view_2_path, "sparse/0")

view_total_path = os.path.join(view_path, "folder_1_2")
view_total_images_path = os.path.join(view_total_path, "images")
view_total_file_path = os.path.join(view_total_path, "sparse/0")

# deal with images
images_1 = os.listdir(view_1_images_path)
images_1.sort(key=lambda x: int(x.split(".")[0]))
images_2 = os.listdir(view_2_images_path)
images_2.sort(key=lambda x: int(x.split(".")[0]))
start_number = int(images_1[0].split(".")[0])
for image in images_1:
    Image.open(os.path.join(view_1_images_path, image)).save(os.path.join(view_total_images_path, f"{start_number}.png"))
    start_number += 1
start_number = int(images_2[0].split(".")[0])
for image in images_2:
    Image.open(os.path.join(view_2_images_path, image)).save(os.path.join(view_total_images_path, f"{start_number}.png"))
    start_number += 1

# deal with cameras.txt
'''
CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
'''
with open(os.path.join(view_1_file_path, "cameras.txt"), "r") as f:
    lines = f.readlines()
    others = lines[0].split(" ")[1:]
with open(os.path.join(view_total_file_path, "cameras.txt"), "w") as f:
    num = 1
    for i in range(len(images_1)):
        f.write(f"{num} {others[0]} {others[1]} {others[2]} {others[3]} {others[4]} {others[5]} {others[6]}")
        num += 1
    for i in range(1, len(images_2)):
        f.write(f"{num} {others[0]} {others[1]} {others[2]} {others[3]} {others[4]} {others[5]} {others[6]}")
        num += 1

# deal with images.txt
'''
IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
POINTS2D[] as (X, Y, POINT3D_ID)  EMPTY
'''
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

with open(os.path.join(view_1_file_path, "images.txt"), "r") as f:
    lines_ori = f.readlines()
    lines = [i for i in lines_ori if i != "\n"]
    lines.sort(key=lambda x: int(x.split(" ")[-1].split(".")[0]))

with open(os.path.join(view_2_file_path, "images.txt"), "r") as f:
    lines_ori = f.readlines()
    lines_2 = [i for i in lines_ori if i != "\n"]
    lines_2.sort(key=lambda x: int(x.split(" ")[-1].split(".")[0]))

with open(os.path.join(view_total_file_path, "images.txt"), "w") as f:
    num = 1
    for i in range(len(images_1)):
        # f.write(lines[i].replace(lines[i].split(" ")[0], str(num)))
        data = lines[i].split(" ")
        data[0] = str(num)
        data[-2] = str(num)
        f.write(" ".join(data)+"\n")
        num += 1
    qw, qx, qy, qz, tx, ty, tz = [float(i) for i in lines[-1].split(" ")[1:8]]
    # Rt[:3, :3] = camera.R.transpose()
    # Rt[:3, 3] = camera.T
    # Rt[3, 3] = 1.0

    # W2C = np.linalg.inv(Rt)
    # pos = W2C[:3, 3]
    # rot = W2C[:3, :3]
    RT_ori = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
                   [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, ty],
                   [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, tz],
                   [0, 0, 0, 1]])
    # print(RT_ori)
    RT_ori = np.linalg.inv(RT_ori)
    RT = dict()
    # print(RT_ori)
    for line in lines_2:
        qw, qx, qy, qz, tx, ty, tz = [float(i) for i in line.split(" ")[1:8]]
        name = line.split(" ")[-1].split(".")[0]
        RT[name] = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
                   [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, ty],
                   [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, tz],
                   [0, 0, 0, 1]])
        RT[name] = np.linalg.inv(RT[name])
    # print("RT 17:\n", RT["17"])
    # # 计算C*B=A
    # C = np.dot(A, np.linalg.pinv(B))
    C = np.dot(RT_ori, np.linalg.pinv(RT["17"]))
    # C = np.dot(RT_ori, RT["17"])
    # print(C)
    # sdf
    for i in range(1, len(images_2)):
        RT_new = np.dot(C, RT[lines_2[i].split(" ")[-1].split(".")[0]])
        RT_new = np.linalg.inv(RT_new)
        qw, qx, qy, qz = R_to_quaternion(RT_new[:3, :3])
        f.write(f'{num} {qw} {qx} {qy} {qz} {RT_new[0, 3]} {RT_new[1, 3]} {RT_new[2, 3]} {num} {lines_2[i].split(" ")[-1]}\n')
        num += 1

# deal with points3D.ply

ply_1 = PlyData.read(os.path.join(view_1_file_path, "points3D.ply"))
ply_2 = PlyData.read(os.path.join(view_2_file_path, "points3D.ply"))
ply_targe = os.path.join(view_total_file_path, "points3D.ply")

data_x_1 = ply_1.elements[0]['x']
data_x_1 = data_x_1.reshape(-1, 1)
data_y_1 = ply_1.elements[0]['y']
data_y_1 = data_y_1.reshape(-1, 1)
data_z_1 = ply_1.elements[0]['z']
data_z_1 = data_z_1.reshape(-1, 1)

ones = np.ones(data_x_1.shape[0]).reshape(-1, 1)
points_1 = np.concatenate((data_x_1, data_y_1, data_z_1, ones), axis=1)

data_x_2 = ply_2.elements[0]['x']
data_x_2 = data_x_2.reshape(-1, 1)
data_y_2 = ply_2.elements[0]['y']
data_y_2 = data_y_2.reshape(-1, 1)
data_z_2 = ply_2.elements[0]['z']
data_z_2 = data_z_2.reshape(-1, 1)

ones = np.ones(data_x_2.shape[0]).reshape(-1, 1)
points_2 = np.concatenate((data_x_2, data_y_2, data_z_2, ones), axis=1)

points_2 = np.dot(C, points_2.T).T
# points_1 = np.dot(C, points_1.T).T
# print(C)

points = np.concatenate((points_1, points_2), axis=0)
data_x = points[:, 0].reshape(-1, 1)
data_y = points[:, 1].reshape(-1, 1)
data_z = points[:, 2].reshape(-1, 1)

points_1_thre = []
for (x, y, z, _) in points_1:
    if (x**2 + y**2 + z**2) < 0.4:
        points_1_thre.append([x, y, z])
# print(np.array(points_1_thre))
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(np.array(points_1_thre))
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(points_2[:, :3])
# trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                             [-0.139, 0.967, -0.215, 0.7],
#                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
C = np.linalg.inv(C)
# C[0, 3] = -0.08
icp_result = o3d.pipelines.registration.registration_icp(
    source, target,
    0.1,  # max_correspondence_distance
    C,  # np.eye(4),  # initial guess transformation matrix
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    # o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # RANSAC
    # o3d.pipelines.registration.TransformationEstimationPointToPlane(False),  # PointToPlane
    # [0.0, 0.0, 0.0],  # initial alignment matrix
    # np.eye(4)  # initial guess transformation matrix
)

# 打印得到的旋转和平移矩阵
print("Transformation matrix:")
print(icp_result.transformation)
# icp_result.transformation[0, 3] = -0.1

# 应用变换到源点云
source.transform(icp_result.transformation)
print(np.asarray(source.points).shape)

# points = np.concatenate((points_1, points_2), axis=0)
points = np.concatenate((np.asarray(target.points), np.asarray(source.points)), axis=0)
print(points.shape)
# data_x = points[:, 0].reshape(-1, 1)
# data_y = points[:, 1].reshape(-1, 1)
# data_z = points[:, 2].reshape(-1, 1)


# 可视化结果
# o3d.visualization.draw_geometries([target, source], window_name="ICP Result")
# o3d.visualization.draw_geometries([target], window_name="ICP Result")


data_nx = np.concatenate((ply_1.elements[0]['nx'], ply_2.elements[0]['nx']), axis=0)
data_nx = data_nx.reshape(-1, 1)

data_ny = np.concatenate((ply_1.elements[0]['ny'], ply_2.elements[0]['ny']), axis=0)
data_ny = data_ny.reshape(-1, 1)

data_nz = np.concatenate((ply_1.elements[0]['nz'], ply_2.elements[0]['nz']), axis=0)
data_nz = data_nz.reshape(-1, 1)

# data_f_dc_0 = np.concatenate((ply_1.elements[0]['f_dc_0'], ply_2.elements[0]['f_dc_0']), axis=0)
# data_f_dc_0 = data_f_dc_0.reshape(-1, 1)

data_red = np.concatenate((ply_1.elements[0]['red'], ply_2.elements[0]['red']), axis=0)
data_red = data_red.reshape(-1, 1)

data_green = np.concatenate((ply_1.elements[0]['green'], ply_2.elements[0]['green']), axis=0)
data_green = data_green.reshape(-1, 1)

data_blue = np.concatenate((ply_1.elements[0]['blue'], ply_2.elements[0]['blue']), axis=0)
data_blue = data_blue.reshape(-1, 1)

data = np.concatenate((data_x, data_y, data_z, data_nx, data_ny, data_nz, data_red, data_green, data_blue), axis=1)
elements = np.empty(data_x.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
elements[:] = list(map(tuple, data))
el = PlyElement.describe(elements, name='vertex')
PlyData([el]).write(ply_targe)