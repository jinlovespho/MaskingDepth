#!/bin/bash

CUDA=3
CONFIG_PATH="./conf/base_train_multiframe_multiselfcrossattn_mask_t.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 