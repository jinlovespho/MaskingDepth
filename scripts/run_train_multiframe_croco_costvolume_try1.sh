#!/bin/bash

CONFIG_PATH="../conf/base_train_multiframe_croco_costvolume_try1.yaml"

CUDA_VISIBLE_DEVICES=1,2 python ../train.py --conf ${CONFIG_PATH} 