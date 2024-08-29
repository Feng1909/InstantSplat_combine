#! /bin/bash

GPU_ID=0
DATA_ROOT_DIR="/home/yugrp01/dataset"
python scripts/gen_multi_folder.py --data_root_dir $DATA_ROOT_DIR --images_num 9 \
                                    --gpu_id $GPU_ID