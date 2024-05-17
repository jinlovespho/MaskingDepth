#!/bin/bash

CUDA=0
CONFIG_PATH="../conf/base_bbox_multiframe_try6.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python ../eval_bbox.py --conf ${CONFIG_PATH} 