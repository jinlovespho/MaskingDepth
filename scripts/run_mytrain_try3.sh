#!/bin/bash

CONFIG_PATH="../conf/my_train_try3.yaml"

CUDA_VISIBLE_DEVICES='2' python ../train.py --conf ${CONFIG_PATH} 