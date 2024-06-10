
DATA_ARGS="
--data_path /media/data1/KITTI 
--dataset kitti_depth
--splits eigen_temp 
--img_ext .jpg 
--re_height 192 
--re_width 640 
"

TRAINING_ARGS="
--num_epoch 30
--batch_size 8
--learning_rate 1e-4
--num_workers 4
--seed 42
"

SAVE_ARGS="
    --epoch_save_freq 3
"

DEPTH_ARGS="
--min_depth 0.1
--max_depth 80.0
"

LOSS_ARGS="
--training_loss supervised_depth
"

MODEL_ARGS="
--model_info mf_try7
--vit_type vit_base
--pretrained_weight croco
--num_prev_frame 1
--cross_attn_depth 4
--masking_ratio 0.0
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240215_MaskingDepth_multiframe
    --wandb_exp_name pho_server5_gpu0123_kitti_bs8_try7_mask00_depth_only_from_ca_map_detach
    --log_path /media/data1/jinlovespho/log/mfdepth
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=0,1,2,3    python ../train.py              \
                                                    ${DATA_ARGS} \
                                                    ${TRAINING_ARGS} \
                                                    ${DEPTH_ARGS} \
                                                    ${LOSS_ARGS} \
                                                    ${MODEL_ARGS} \
                                                    ${SAVE_ARGS} \
                                                    ${LOGGING_ARGS} \
                                                    ${ETC_ARGS} \
