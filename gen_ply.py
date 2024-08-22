from plyfile import PlyData, PlyElement
import numpy as np
# from mayavi import mlab
 
# 读取ply文件，并展示3d模型
ply_1 = PlyData.read("/home/yugrp01/InstantSplat/output/infer/self/mini/9_views_1000Iter_1xPoseLR/point_cloud/iteration_1000/point_cloud_1.ply")
ply_2 = PlyData.read("/home/yugrp01/InstantSplat/output/infer/self/mini/9_views_1000Iter_1xPoseLR/point_cloud/iteration_1000/point_cloud_2.ply")
target_ply = "/home/yugrp01/InstantSplat/output/infer/self/mini/9_views_1000Iter_1xPoseLR/point_cloud/iteration_1000/point_cloud.ply"

# data_x = np.concatenate((ply_1.elements[0]['x'], ply_2.elements[0]['x']), axis=0)
# data_x = data_x.reshape(-1, 1)

# data_y = np.concatenate((ply_1.elements[0]['y'], ply_2.elements[0]['y']), axis=0)
# data_y = data_y.reshape(-1, 1)

# data_z = np.concatenate((ply_1.elements[0]['z'], ply_2.elements[0]['z']), axis=0)
# data_z = data_z.reshape(-1, 1)
data_x_1 = ply_1.elements[0]['x']
print(data_x_1)
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

# [[ 6.44667372e-01 -1.26907194e-01 -7.53855786e-01 -9.68720654e-02]
#  [ 2.95271260e-02  9.89522122e-01 -1.41329818e-01 -1.27461710e-02]
#  [ 7.63892748e-01  6.88515267e-02  6.41659831e-01  6.70255455e-02]
#  [ 4.16333634e-17  4.16333634e-17 -2.77555756e-16  1.00000000e+00]]
C = np.array([[0.644667372, -0.126907194, -0.753855786, -0.0968720654],
                [0.029527126, 0.989522122, -0.141329818, -0.0127461710],
                [0.763892748, 0.0688515267, 0.641659831, 0.0670255455],
                [0, 0, 0, 1]])
# C = np.array([[1, 0, 0, 0],
#               [0, 1, 0, 0],
#               [0, 0, 1, 0],
#               [0, 0, 0, 1]])
points_2 = np.dot(C, points_2.T).T
# print(points_next)
points = np.concatenate((points_1, points_2), axis=0)
print(points)
data_x = points[:, 0].reshape(-1, 1)
data_y = points[:, 1].reshape(-1, 1)
data_z = points[:, 2].reshape(-1, 1)
# sdf

data_nx = np.concatenate((ply_1.elements[0]['nx'], ply_2.elements[0]['nx']), axis=0)
data_nx = data_nx.reshape(-1, 1)

data_ny = np.concatenate((ply_1.elements[0]['ny'], ply_2.elements[0]['ny']), axis=0)
data_ny = data_ny.reshape(-1, 1)

data_nz = np.concatenate((ply_1.elements[0]['nz'], ply_2.elements[0]['nz']), axis=0)
data_nz = data_nz.reshape(-1, 1)

data_f_dc_0 = np.concatenate((ply_1.elements[0]['f_dc_0'], ply_2.elements[0]['f_dc_0']), axis=0)
data_f_dc_0 = data_f_dc_0.reshape(-1, 1)

data_f_dc_1 = np.concatenate((ply_1.elements[0]['f_dc_1'], ply_2.elements[0]['f_dc_1']), axis=0)
data_f_dc_1 = data_f_dc_1.reshape(-1, 1)

data_f_dc_2 = np.concatenate((ply_1.elements[0]['f_dc_2'], ply_2.elements[0]['f_dc_2']), axis=0)
data_f_dc_2 = data_f_dc_2.reshape(-1, 1)

data_f_rest_0 = np.concatenate((ply_1.elements[0]['f_rest_0'], ply_2.elements[0]['f_rest_0']), axis=0)
data_f_rest_0 = data_f_rest_0.reshape(-1, 1)

data_f_rest_1 = np.concatenate((ply_1.elements[0]['f_rest_1'], ply_2.elements[0]['f_rest_1']), axis=0)
data_f_rest_1 = data_f_rest_1.reshape(-1, 1)

data_f_rest_2 = np.concatenate((ply_1.elements[0]['f_rest_2'], ply_2.elements[0]['f_rest_2']), axis=0)
data_f_rest_2 = data_f_rest_2.reshape(-1, 1)

data_f_rest_3 = np.concatenate((ply_1.elements[0]['f_rest_3'], ply_2.elements[0]['f_rest_3']), axis=0)
data_f_rest_3 = data_f_rest_3.reshape(-1, 1)

data_f_rest_4 = np.concatenate((ply_1.elements[0]['f_rest_4'], ply_2.elements[0]['f_rest_4']), axis=0)
data_f_rest_4 = data_f_rest_4.reshape(-1, 1)

data_f_rest_5 = np.concatenate((ply_1.elements[0]['f_rest_5'], ply_2.elements[0]['f_rest_5']), axis=0)
data_f_rest_5 = data_f_rest_5.reshape(-1, 1)

data_f_rest_6 = np.concatenate((ply_1.elements[0]['f_rest_6'], ply_2.elements[0]['f_rest_6']), axis=0)
data_f_rest_6 = data_f_rest_6.reshape(-1, 1)

data_f_rest_7 = np.concatenate((ply_1.elements[0]['f_rest_7'], ply_2.elements[0]['f_rest_7']), axis=0)
data_f_rest_7 = data_f_rest_7.reshape(-1, 1)

data_f_rest_8 = np.concatenate((ply_1.elements[0]['f_rest_8'], ply_2.elements[0]['f_rest_8']), axis=0)
data_f_rest_8 = data_f_rest_8.reshape(-1, 1)

data_f_rest_9 = np.concatenate((ply_1.elements[0]['f_rest_9'], ply_2.elements[0]['f_rest_9']), axis=0)
data_f_rest_9 = data_f_rest_9.reshape(-1, 1)

data_f_rest_10 = np.concatenate((ply_1.elements[0]['f_rest_10'], ply_2.elements[0]['f_rest_10']), axis=0)
data_f_rest_10 = data_f_rest_10.reshape(-1, 1)

data_f_rest_11 = np.concatenate((ply_1.elements[0]['f_rest_11'], ply_2.elements[0]['f_rest_11']), axis=0)
data_f_rest_11 = data_f_rest_11.reshape(-1, 1)

data_f_rest_12 = np.concatenate((ply_1.elements[0]['f_rest_12'], ply_2.elements[0]['f_rest_12']), axis=0)
data_f_rest_12 = data_f_rest_12.reshape(-1, 1)

data_f_rest_13 = np.concatenate((ply_1.elements[0]['f_rest_13'], ply_2.elements[0]['f_rest_13']), axis=0)
data_f_rest_13 = data_f_rest_13.reshape(-1, 1)

data_f_rest_14 = np.concatenate((ply_1.elements[0]['f_rest_14'], ply_2.elements[0]['f_rest_14']), axis=0)
data_f_rest_14 = data_f_rest_14.reshape(-1, 1)

data_f_rest_15 = np.concatenate((ply_1.elements[0]['f_rest_15'], ply_2.elements[0]['f_rest_15']), axis=0)
data_f_rest_15 = data_f_rest_15.reshape(-1, 1)

data_f_rest_16 = np.concatenate((ply_1.elements[0]['f_rest_16'], ply_2.elements[0]['f_rest_16']), axis=0)
data_f_rest_16 = data_f_rest_16.reshape(-1, 1)

data_f_rest_17 = np.concatenate((ply_1.elements[0]['f_rest_17'], ply_2.elements[0]['f_rest_17']), axis=0)
data_f_rest_17 = data_f_rest_17.reshape(-1, 1)

data_f_rest_18 = np.concatenate((ply_1.elements[0]['f_rest_18'], ply_2.elements[0]['f_rest_18']), axis=0)
data_f_rest_18 = data_f_rest_18.reshape(-1, 1)

data_f_rest_19 = np.concatenate((ply_1.elements[0]['f_rest_19'], ply_2.elements[0]['f_rest_19']), axis=0)
data_f_rest_19 = data_f_rest_19.reshape(-1, 1)

data_f_rest_20 = np.concatenate((ply_1.elements[0]['f_rest_20'], ply_2.elements[0]['f_rest_20']), axis=0)
data_f_rest_20 = data_f_rest_20.reshape(-1, 1)

data_f_rest_21 = np.concatenate((ply_1.elements[0]['f_rest_21'], ply_2.elements[0]['f_rest_21']), axis=0)
data_f_rest_21 = data_f_rest_21.reshape(-1, 1)

data_f_rest_22 = np.concatenate((ply_1.elements[0]['f_rest_22'], ply_2.elements[0]['f_rest_22']), axis=0)
data_f_rest_22 = data_f_rest_22.reshape(-1, 1)

data_f_rest_23 = np.concatenate((ply_1.elements[0]['f_rest_23'], ply_2.elements[0]['f_rest_23']), axis=0)
data_f_rest_23 = data_f_rest_23.reshape(-1, 1)

data_f_rest_24 = np.concatenate((ply_1.elements[0]['f_rest_24'], ply_2.elements[0]['f_rest_24']), axis=0)
data_f_rest_24 = data_f_rest_24.reshape(-1, 1)

data_f_rest_25 = np.concatenate((ply_1.elements[0]['f_rest_25'], ply_2.elements[0]['f_rest_25']), axis=0)
data_f_rest_25 = data_f_rest_25.reshape(-1, 1)

data_f_rest_26 = np.concatenate((ply_1.elements[0]['f_rest_26'], ply_2.elements[0]['f_rest_26']), axis=0)
data_f_rest_26 = data_f_rest_26.reshape(-1, 1)

data_f_rest_27 = np.concatenate((ply_1.elements[0]['f_rest_27'], ply_2.elements[0]['f_rest_27']), axis=0)
data_f_rest_27 = data_f_rest_27.reshape(-1, 1)

data_f_rest_28 = np.concatenate((ply_1.elements[0]['f_rest_28'], ply_2.elements[0]['f_rest_28']), axis=0)
data_f_rest_28 = data_f_rest_28.reshape(-1, 1)

data_f_rest_29 = np.concatenate((ply_1.elements[0]['f_rest_29'], ply_2.elements[0]['f_rest_29']), axis=0)
data_f_rest_29 = data_f_rest_29.reshape(-1, 1)

data_f_rest_30 = np.concatenate((ply_1.elements[0]['f_rest_30'], ply_2.elements[0]['f_rest_30']), axis=0)
data_f_rest_30 = data_f_rest_30.reshape(-1, 1)

data_f_rest_31 = np.concatenate((ply_1.elements[0]['f_rest_31'], ply_2.elements[0]['f_rest_31']), axis=0)
data_f_rest_31 = data_f_rest_31.reshape(-1, 1)

data_f_rest_32 = np.concatenate((ply_1.elements[0]['f_rest_32'], ply_2.elements[0]['f_rest_32']), axis=0)
data_f_rest_32 = data_f_rest_32.reshape(-1, 1)

data_f_rest_33 = np.concatenate((ply_1.elements[0]['f_rest_33'], ply_2.elements[0]['f_rest_33']), axis=0)
data_f_rest_33 = data_f_rest_33.reshape(-1, 1)

data_f_rest_34 = np.concatenate((ply_1.elements[0]['f_rest_34'], ply_2.elements[0]['f_rest_34']), axis=0)
data_f_rest_34 = data_f_rest_34.reshape(-1, 1)

data_f_rest_35 = np.concatenate((ply_1.elements[0]['f_rest_35'], ply_2.elements[0]['f_rest_35']), axis=0)
data_f_rest_35 = data_f_rest_35.reshape(-1, 1)

data_f_rest_36 = np.concatenate((ply_1.elements[0]['f_rest_36'], ply_2.elements[0]['f_rest_36']), axis=0)
data_f_rest_36 = data_f_rest_36.reshape(-1, 1)

data_f_rest_37 = np.concatenate((ply_1.elements[0]['f_rest_37'], ply_2.elements[0]['f_rest_37']), axis=0)
data_f_rest_37 = data_f_rest_37.reshape(-1, 1)

data_f_rest_38 = np.concatenate((ply_1.elements[0]['f_rest_38'], ply_2.elements[0]['f_rest_38']), axis=0)
data_f_rest_38 = data_f_rest_38.reshape(-1, 1)

data_f_rest_39 = np.concatenate((ply_1.elements[0]['f_rest_39'], ply_2.elements[0]['f_rest_39']), axis=0)
data_f_rest_39 = data_f_rest_39.reshape(-1, 1)

data_f_rest_40 = np.concatenate((ply_1.elements[0]['f_rest_40'], ply_2.elements[0]['f_rest_40']), axis=0)
data_f_rest_40 = data_f_rest_40.reshape(-1, 1)

data_f_rest_41 = np.concatenate((ply_1.elements[0]['f_rest_41'], ply_2.elements[0]['f_rest_41']), axis=0)
data_f_rest_41 = data_f_rest_41.reshape(-1, 1)

data_f_rest_42 = np.concatenate((ply_1.elements[0]['f_rest_42'], ply_2.elements[0]['f_rest_42']), axis=0)
data_f_rest_42 = data_f_rest_42.reshape(-1, 1)

data_f_rest_43 = np.concatenate((ply_1.elements[0]['f_rest_43'], ply_2.elements[0]['f_rest_43']), axis=0)
data_f_rest_43 = data_f_rest_43.reshape(-1, 1)

data_f_rest_44 = np.concatenate((ply_1.elements[0]['f_rest_44'], ply_2.elements[0]['f_rest_44']), axis=0)
data_f_rest_44 = data_f_rest_44.reshape(-1, 1)

data_opacity = np.concatenate((ply_1.elements[0]['opacity'], ply_2.elements[0]['opacity']), axis=0)
data_opacity = data_opacity.reshape(-1, 1)

data_scale_0 = np.concatenate((ply_1.elements[0]['scale_0'], ply_2.elements[0]['scale_0']), axis=0)
data_scale_0 = data_scale_0.reshape(-1, 1)

data_scale_1 = np.concatenate((ply_1.elements[0]['scale_1'], ply_2.elements[0]['scale_1']), axis=0)
data_scale_1 = data_scale_1.reshape(-1, 1)

data_scale_2 = np.concatenate((ply_1.elements[0]['scale_2'], ply_2.elements[0]['scale_2']), axis=0)
data_scale_2 = data_scale_2.reshape(-1, 1)

data_rot_0 = np.concatenate((ply_1.elements[0]['rot_0'], ply_2.elements[0]['rot_0']), axis=0)
data_rot_0 = data_rot_0.reshape(-1, 1)

data_rot_1 = np.concatenate((ply_1.elements[0]['rot_1'], ply_2.elements[0]['rot_1']), axis=0)
data_rot_1 = data_rot_1.reshape(-1, 1)

data_rot_2 = np.concatenate((ply_1.elements[0]['rot_2'], ply_2.elements[0]['rot_2']), axis=0)
data_rot_2 = data_rot_2.reshape(-1, 1)

data_rot_3 = np.concatenate((ply_1.elements[0]['rot_3'], ply_2.elements[0]['rot_3']), axis=0)
data_rot_3 = data_rot_3.reshape(-1, 1)

data = np.concatenate((data_x, data_y, data_z, data_nx, data_ny, data_nz, data_f_dc_0, data_f_dc_1, data_f_dc_2, data_f_rest_0, data_f_rest_1, data_f_rest_2, data_f_rest_3, data_f_rest_4, data_f_rest_5, data_f_rest_6, data_f_rest_7, data_f_rest_8, data_f_rest_9, data_f_rest_10, data_f_rest_11, data_f_rest_12, data_f_rest_13, data_f_rest_14, data_f_rest_15, data_f_rest_16, data_f_rest_17, data_f_rest_18, data_f_rest_19, data_f_rest_20, data_f_rest_21, data_f_rest_22, data_f_rest_23, data_f_rest_24, data_f_rest_25, data_f_rest_26, data_f_rest_27, data_f_rest_28, data_f_rest_29, data_f_rest_30, data_f_rest_31, data_f_rest_32, data_f_rest_33, data_f_rest_34, data_f_rest_35, data_f_rest_36, data_f_rest_37, data_f_rest_38, data_f_rest_39, data_f_rest_40, data_f_rest_41, data_f_rest_42, data_f_rest_43, data_f_rest_44, data_opacity, data_scale_0, data_scale_1, data_scale_2, data_rot_0, data_rot_1, data_rot_2, data_rot_3), axis=1)

# data = np.concatenate((data_x, data_y), axis=1)

# elements = np.empty(data_x.shape[0], dtype=[('rot_2', 'f4'), ('rot_3', 'f4')])
elements = np.empty(data_x.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), 
                                            ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'), ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'), ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'), ('f_rest_9', 'f4'), ('f_rest_10', 'f4'), ('f_rest_11', 'f4'), ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'), ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'), ('f_rest_18', 'f4'), ('f_rest_19', 'f4'), ('f_rest_20', 'f4'), ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'), ('f_rest_24', 'f4'), ('f_rest_25', 'f4'), ('f_rest_26', 'f4'), ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'), ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'), ('f_rest_33', 'f4'), ('f_rest_34', 'f4'), ('f_rest_35', 'f4'), ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'), ('f_rest_39', 'f4'), ('f_rest_40', 'f4'), ('f_rest_41', 'f4'), ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'), 
                                            ('opacity', 'f4'), ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])
elements[:] = list(map(tuple, data))
print(elements)

el = PlyElement.describe(elements, name='vertex')

# data_2 = np.concatenate((ply_1.elements[0]['rot_3'], ply_2.elements[0]['rot_3']), axis=0)
# data_2 = np.array(data_2, dtype=[('rot_3', 'f4')])
# data_2 = PlyElement.describe(data_2, name='vertex1')
# elements = np.empty(data_2.shape[0], dtype=dtype_full)
# elements[:] = list(map(tuple, data))
# el = PlyElement.describe(elements, 'vertex')
# PlyData([el]).write(path)

# PlyData([data_1]).write(target_ply)
PlyData([el]).write(target_ply)
# print("=============")
# ply = PlyData.read(target_ply)
# print(ply)
# print(ply.elements[0].data['rot_2'])