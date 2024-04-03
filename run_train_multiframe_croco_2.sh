#!/bin/bash

CUDA=5
CONFIG_PATH="./conf/base_train_multiframe_croco_2.yaml"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 