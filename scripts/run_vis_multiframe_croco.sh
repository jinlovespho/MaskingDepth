#!/bin/bash

CUDA=7
CONFIG_PATH="../conf/base_vis_multiframe_croco.yaml"

CUDA_VISIBLE_DEVICES=${CUDA} python ../visualize.py --conf ${CONFIG_PATH} 