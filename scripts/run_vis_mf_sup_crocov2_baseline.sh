
DATA_ARGS="
    --data_path /media/data1/KITTI 
    --dataset kitti_depth
    --splits eigen_temp 
    --img_ext .jpg 
    --re_height 192 
    --re_width 640 
"

TRAINING_ARGS="
    --batch_size 16
    --num_workers 4
    --seed 42
"

DEPTH_ARGS="
    --min_depth 0.1
    --max_depth 80.0
"

LOSS_ARGS="
    --training_loss supervised_depth
"

# crocostereo.pth
# CroCo_V2_ViTBase_SmallDecoder.pth
# CroCo_V2_ViTBase_BaseDecoder.pth
# CroCo_V2_ViTLarge_BaseDecoder.pth

MODEL_ARGS="
    --model_info vis_mf_sup_crocov2_baseline
    --pretrained_weight_path ../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth
    --load_weight_path /media/data1/jinlovespho/log/mfdepth/pho_server5_gpu3_kitti_bs8_mf_sup_baseline_crocov2_encB_decB/weights_50/depth.pth
"

SAVE_ARGS="
    --epoch_save_freq 10
"

LOGGING_ARGS="
    --log_tool wandb
    --wandb_proj_name 20240612_MultiFrame_Depth
    --wandb_exp_name vis_pho_server5_kitti_bs16_mf_sup_baseline_crocov2_encB_decB
    --log_path /media/data1/jinlovespho/log/mfdepth
"

ETC_ARGS="
    
"


CUDA_VISIBLE_DEVICES=0 python ../vis.py     ${DATA_ARGS} \
                                            ${TRAINING_ARGS} \
                                            ${DEPTH_ARGS} \
                                            ${LOSS_ARGS} \
                                            ${MODEL_ARGS} \
                                            ${SAVE_ARGS} \
                                            ${LOGGING_ARGS} \
                                            ${ETC_ARGS} \
