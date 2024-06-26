
DATA_ARGS="
    --data_path /media/data1/KITTI 
    --dataset kitti_depth
    --splits eigen_temp 
    --img_ext .jpg 
    --re_height 192 
    --re_width 640 
"

TRAINING_ARGS="
    --num_epoch 50
    --batch_size 8
    --backbone_lr 1e-5
    --lr 1e-4
    --num_workers 4
    --seed 42
"

DEPTH_ARGS="
    --min_depth 0.1
    --max_depth 80.0
"

LOSS_ARGS="
    --training_loss supervised_depth
"

MODEL_ARGS="
    --model_info mf_sup_vit_baseline
    --pretrained_weight_path ../pretrained_weights/vit_base_384.pth
    --vit_type vit_base
    --pretrained_weight vit_base_384
    --masking_ratio 0.0
    --cross_attn_depth 4
    --num_prev_frame 1
"

SAVE_ARGS="
    --epoch_save_freq 25
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240612_MultiFrame_Depth
    --wandb_exp_name pho_server5_gpu3_kitti_bs8_mf_sup_baseline_vitB
    --log_path /media/data1/jinlovespho/log/mfdepth
"

ETC_ARGS="
    
"

CUDA_VISIBLE_DEVICES=3 python ../train.py   ${DATA_ARGS} \
                                            ${TRAINING_ARGS} \
                                            ${DEPTH_ARGS} \
                                            ${LOSS_ARGS} \
                                            ${MODEL_ARGS} \
                                            ${SAVE_ARGS} \
                                            ${LOGGING_ARGS} \
                                            ${ETC_ARGS} \
