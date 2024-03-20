#!/bin/bash

CUDA=3
CONFIG_PATH="./conf/base_train.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python train.py --conf ${CONFIG_PATH} 