#!/bin/bash

CUDA=7
CONFIG_PATH="./conf/base_train_multiframe_croco.yaml"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 