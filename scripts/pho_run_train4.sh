

DATA_ARGS="
    --data_path /media/dataset2/KITTI
    --dataset kitti
    --splits eigen_zhou
    --img_ext .jpg 
    --re_height 192 
    --re_width 640 
"


TRAINING_ARGS="
    --num_epoch 20
    --batch_size 4
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
    --zero_aug 0.5
    --moving_masking no_grad
    --attn_agg_tf
    --attn_agg_tf_multiscale
    --attn_agg_tf_cat_out_and_map
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name pho_server11_gpu3_kitti_croco_attnaggtest_zero05_try4_attntf_multiscale_catMapAndOut
    --log_path /media/dataset1/jinlovespho/aaai_log
"


ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=3   python ./train.py        \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${SAVE_ARGS} \
                                                ${LOGGING_ARGS} \
                                                ${ETC_ARGS} \
