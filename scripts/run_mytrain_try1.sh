#!/bin/bash

CONFIG_PATH="../conf/my_train_try1.yaml"

CUDA_VISIBLE_DEVICES='3' python ../train.py --conf ${CONFIG_PATH} 