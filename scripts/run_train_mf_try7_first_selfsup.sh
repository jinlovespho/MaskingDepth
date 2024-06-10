
DATA_ARGS="
--data_path /media/data1/KITTI 
--dataset kitti
--splits eigen_zhou 
--img_ext .jpg 
--re_height 192 
--re_width 640 
"

TRAINING_ARGS="
--num_epoch 50
--batch_size 8
--learning_rate 1e-4
--num_workers 4
--seed 42
"

DEPTH_ARGS="
--min_depth 0.1
--max_depth 80.0
"

LOSS_ARGS="
--training_loss selfsupervised_img_recon
"

MODEL_ARGS="
--model_info mf_try7
--vit_type vit_base
--pretrained_weight croco
--num_prev_frame 1
--cross_attn_depth 4
--masking_ratio 0.8
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --wandb
    --wandb_proj_name 20240215_MaskingDepth_multiframe
    --wandb_exp_name pho_server05_gpu3_kitti_bs8_try7_mask08
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
