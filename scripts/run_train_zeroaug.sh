

DATA_ARGS="
--data_path /home/cvlab08/projects/data/KITTI
--dataset kitti
--splits eigen_zhou
--img_ext .jpg 
--re_height 192 
--re_width 640 
"


TRAINING_ARGS="
--num_epoch 20
--batch_size 8
--learning_rate 1e-4
--num_workers 4
--seed 41
"

DEPTH_ARGS="
--min_depth 0.1
--max_depth 100.0
"

LOSS_ARGS="
--training_loss selfsupervised_img_recon
--use_future_frame
"

MODEL_ARGS="
--model_info croco
--vit_type vit_base
--pretrained_weight vit_base_384
--pretrained_path ./CroCo_V2_ViTBase_BaseDecoder.pth
--zero_aug 0.33
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240612_MultiFrame_Depth
    --wandb_exp_name hg_croco_multi_zeroaug
    --log_path /home/cvlab08/projects/data/hg_log/selfsup_depth
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=1   python ./train.py        \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${SAVE_ARGS} \
                                                ${LOGGING_ARGS} \
                                                ${ETC_ARGS} \
