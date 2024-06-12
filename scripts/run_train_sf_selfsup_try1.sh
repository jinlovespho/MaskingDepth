

DATA_ARGS="
--data_path /home/cvlab08/projects/data/KITTI
--dataset kitti_depth
--splits eigen_zhou
--img_ext .jpg 
--re_height 192 
--re_width 640 
"


TRAINING_ARGS="
--num_epoch 50
--batch_size 16
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
--use_future_frame
"

MODEL_ARGS="
--model_info sf_selfsup_try1
--vit_type vit_base
--pretrained_weight vit_base_384
"

SAVE_ARGS="
    --epoch_save_freq 10
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240612_MultiFrame_Depth
    --wandb_exp_name pho_server8_gpu1_kitti_bs16_sf_selfsup_try1_eigenzhou
    --log_path /home/cvlab08/projects/data/jinlovespho/log/mfdepth
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=1   python ../train.py        \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${SAVE_ARGS} \
                                                ${LOGGING_ARGS} \
                                                ${ETC_ARGS} \
