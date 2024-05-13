#!/bin/bash

CUDA='0,1,2'
CONFIG_PATH="../conf/base_train_singleframe_baseline.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python ../train.py --conf ${CONFIG_PATH}