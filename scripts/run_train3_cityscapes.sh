

DATA_ARGS="
    --data_path /media/dataset2/cityscapes/cityscapes_preprocessed
    --dataset cityscapes
    --splits cityscapes
    --cs_val_path /media/dataset2/cityscapes
    --cs_gt_path /media/dataset2/cityscapes/gt_depths
    --img_ext .jpg 
    --re_height 192
    --re_width 512
"

# cs resol (128,416) or (192,512)


TRAINING_ARGS="
--num_epoch 20
--batch_size 10
--learning_rate 1e-4
--num_workers 0
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
"

SAVE_ARGS="
    --epoch_save_freq 5
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name pho_server5_gpu3_CITYSCAPES_croco_basebase_attnaggtest_zero05_topk_tf_lr1e4_fix_res192_512
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
