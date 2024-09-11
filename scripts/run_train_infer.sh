#! /bin/bash

GPU_ID=0
# DATA_ROOT_DIR="/ssd2/zhiwen/projects/InstantSplat/data"
DATA_ROOT_DIR="/home/yugrp01/Downloads/instantsplat_train_test_split/collated_instantsplat_data/eval"
DATASETS=(
    # TT
    # sora
    # mars
    # Mipnerf
    # Tanks
    self
    )

SCENES=(
    # Family
    # Barn
    # Francis
    # Horse
    # Ignatius
    # santorini
    # bicycle
    # Ballroom
    mini
    )

N_VIEWS=(
    # 3
    # 5
    # 9
    # 10
    # 12
    # 14
    17
    # 168
    # 20
    # 24
    )

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=10
pose_lr=1x

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            # SOURCE_PATH must be Absolute path
            SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views
            MODEL_PATH=./output/infer/${DATASET}/${SCENE}/${N_VIEW}_views_${gs_train_iter}Iter_${pose_lr}PoseLR/

            # # ----- (1) Dust3r_coarse_geometric_initialization -----
            CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./coarse_init_infer.py \
            --img_base_path ${SOURCE_PATH} \
            --n_views ${N_VIEW}  \
            --focal_avg \
            --focal 351.6702 \
            "

            # # ----- (2) Train: jointly optimize pose -----
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./train_joint.py \
            -s /home/yugrp01/dataset/folder_1_2 \
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
            # eval $CMD_D1
            echo "========= ${SCENE}: Train: jointly optimize pose ========="
            eval $CMD_T
            echo "========= ${SCENE}: Render interpolated pose & output video ========="
            eval $CMD_RI
            done
        done
    done