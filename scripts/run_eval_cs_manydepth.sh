
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
    --ckpt_name cs_manydepth
    --ckpt_path /home/cvlab05/project/jinlovespho/github/monodepth/pho_cs/MaskingDepth/pretrained_weights/manydepth/CityScapes_MR
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

MODEL_ARGS="
    --model_info cs_manydepth
    --pretrained_path ./pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth
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

 