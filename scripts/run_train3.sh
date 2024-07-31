

DATA_ARGS="
    --data_path /media/data1/KITTI
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
    --smooth_weight 1e-3
"

MODEL_ARGS="
    --model_info croco
    --vit_type vit_base
    --pretrained_weight vit_base_384
    --pretrained_path ./pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth
    --attn_agg
    --softmax_attn
    --fwd_pass 2
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name pho_server5_gpu0123_kitti_croco_try3_attnaggtest_distill
    --log_path /media/dataset1/jinlovespho/aaai_log
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=0,1,2,3   python ./train.py        \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${SAVE_ARGS} \
                                                ${LOGGING_ARGS} \
                                                ${ETC_ARGS} \
