#!/bin/bash

CUDA=0
CONFIG_PATH="./conf/base_train_multiframe.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 