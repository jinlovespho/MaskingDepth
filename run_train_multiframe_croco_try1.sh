#!/bin/bash

CUDA=2
CONFIG_PATH="./conf/base_train_multiframe_croco_try1.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 