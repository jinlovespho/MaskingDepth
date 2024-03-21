#!/bin/bash

CUDA=1
CONFIG_PATH="./conf/base_train_multiframe_croco_3.yaml"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 