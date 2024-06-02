#!/bin/bash

CONFIG_PATH="../conf/my_train_try2.yaml"

CUDA_VISIBLE_DEVICES='1' python ../train.py --conf ${CONFIG_PATH} 