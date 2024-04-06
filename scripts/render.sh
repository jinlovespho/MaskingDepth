#!/bin/bash

CUDA=3
CONFIG_PATH="./conf/base_train_save_depth.yaml"


CUDA_VISIBLE_DEVICES=${CUDA} python save_train_depth.py --conf ${CONFIG_PATH} 