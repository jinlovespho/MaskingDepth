
# NOT USED! 

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


EVAL_ARGS="
    --eval
    --ckpt_name none
    --ckpt_path none
    --batch_size 1
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

# ./pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
# ./pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
# ./pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth

MODEL_ARGS="
    --model_info cs_dust3r
    --pretrained_path ./pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
    --attn_agg
    --softmax_attn
    --zero_aug 0.5
    --moving_masking no_grad_topk
    --attn_agg_tf
"


LOGGING_ARGS="
    --log_tool wandba
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name SUPPL_EVAL_pho_server5_gpu0_CITYSCAPES_croco_basebase_attnaggtest_zero05_topk_tf_lr3e5_fix_res192_512_mskSameMed
    --log_path /media/dataset1/jinlovespho/aaai_log
"




CUDA_VISIBLE_DEVICES=0   python ./eval_cs.py        \
                                                ${DATA_ARGS} \
                                                ${EVAL_ARGS} \
                                                ${DEPTH_ARGS} \
                                                ${LOSS_ARGS} \
                                                ${MODEL_ARGS} \
                                                ${LOGGING_ARGS} \

 