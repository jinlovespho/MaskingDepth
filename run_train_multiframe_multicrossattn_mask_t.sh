#!/bin/bash

CUDA=0
CONFIG_PATH="./conf/base_train_multiframe_multicrossattn_mask_t.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 