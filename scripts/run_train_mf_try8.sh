
DATA_ARGS="
--data_path /home/cvlab08/projects/data/KITTI
--dataset kitti_depth
--splits eigen_temp 
--img_ext .jpg 
--re_height 192 
--re_width 640 
"

TRAINING_ARGS="
--num_epoch 50
--batch_size 2
--learning_rate 1e-4
--num_workers 4
--seed 42
"

SAVE_ARGS="
    --epoch_save_freq 10
"

DEPTH_ARGS="
--min_depth 0.1
--max_depth 80.0
"

LOSS_ARGS="
--training_loss supervised_depth
"

MODEL_ARGS="
--model_info mf_try8
--vit_type vit_base
--pretrained_weight croco
--num_prev_frame 1
--cross_attn_depth 4
--masking_ratio 0.0
"

LOGGING_ARGS="
    --log_tool asdf
    --wandb_proj_name 20240215_MaskingDepth_multiframe
    --wandb_exp_name pho_server8_gpu0123_kitti_bs8_try8_mask08_depth_only_from_ca_map_detach
    --log_path /home/cvlab08/projects/data/jinlovespho/log/mfdepth
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=0  python ../train.py              \
                                                    ${DATA_ARGS} \
                                                    ${TRAINING_ARGS} \
                                                    ${DEPTH_ARGS} \
                                                    ${LOSS_ARGS} \
                                                    ${MODEL_ARGS} \
                                                    ${SAVE_ARGS} \
                                                    ${LOGGING_ARGS} \
                                                    ${ETC_ARGS} \
