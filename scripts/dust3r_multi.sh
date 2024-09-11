#! /bin/bash

GPU_ID=0
DATASET=self

SCENE=mini

N_VIEW=139
gs_train_iter=3000
pose_lr=1x
MODEL_PATH=./output/infer/${DATASET}/${SCENE}/${N_VIEW}_views_${gs_train_iter}Iter_${pose_lr}PoseLR/

# # ----- (2) Train: jointly optimize pose -----
CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_joint.py \
-s /home/yugrp01/dataset/folder_all \
-m ${MODEL_PATH}  \
--n_views ${N_VIEW}  \
--scene ${SCENE} \
--iter ${gs_train_iter} \
--optim_pose \
"

# ----- (3) Render interpolated pose & output video -----
CMD_RI="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./render_by_interp.py \
-s /home/yugrp01/dataset/folder_all \
-m ${MODEL_PATH}  \
--n_views ${N_VIEW}  \
--scene ${SCENE} \
--iter ${gs_train_iter} \
--eval \
--get_video \
"


# echo "========= ${SCENE}: Dust3r_coarse_geometric_initialization ========="
# python scripts/gen_multi_folder.py --data_root_dir "/home/yugrp01/dataset" --images_num 9 \
#                                     --gpu_id $GPU_ID
echo "========= ${SCENE}: Train: jointly optimize pose ========="
eval $CMD_T
echo "========= ${SCENE}: Render interpolated pose & output video ========="
eval $CMD_RI