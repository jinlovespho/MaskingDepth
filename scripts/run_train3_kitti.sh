

DATA_ARGS="
    --data_path /media/data1/KITTI
    --dataset kitti
    --splits eigen_zhou
    --img_ext .jpg 
    --re_height 192 
    --re_width 640 
"


TRAINING_ARGS="
    --num_epoch 30
    --batch_size 1
    --learning_rate 5e-5
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
    --moving_masking no_grad_topk
    --attn_agg_tf
    --encoder_freeze
    --decoder_freeze
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --log_tool wandba
    --wandb_proj_name 20240612_MultiFrame_Depth
    --wandb_exp_name SUPPL_pho_server5_gpu0_kitti_croco_basebase_attnaggtest_zero05_topk_tf_lr5e5_fix_freezeEncDec
    --log_path /media/dataset1/jinlovespho/aaai_log
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=0   python ./train.py        \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${SAVE_ARGS} \
                                                ${LOGGING_ARGS} \
                                                ${ETC_ARGS} \
