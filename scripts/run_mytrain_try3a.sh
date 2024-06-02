#!/bin/bash

CONFIG_PATH="../conf/my_train_try3a.yaml"

CUDA_VISIBLE_DEVICES='2' python ../train.py --conf ${CONFIG_PATH} 