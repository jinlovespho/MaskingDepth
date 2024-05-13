#!/bin/bash

CONFIG_PATH="./conf/base_train_multiframe_croco_try6.yaml"

CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --conf ${CONFIG_PATH} 