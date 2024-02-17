#!/bin/bash


CONFIG_PATH="./conf/base_train.yaml"

CUDA_VISIBLE_DEVICES=0 python train.py --conf ${CONFIG_PATH} 