#!/bin/bash

CUDA=4
CONFIG_PATH="./conf/base_train_multiframe_croco_4.yaml"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 