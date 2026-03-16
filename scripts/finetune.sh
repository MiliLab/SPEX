export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=4,5,6,7

LLM_VERSION="./checkpoints/Qwen2_5_1_5B_mask_decoder"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="./spex/pertrained_weights/InternImage"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

NUM_GPUS=$1
NNODES=1
RANK=0
ADDR="127.0.0.1"
PORT=49500
PROMPT_VERSION="qwen_2_5"
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ./cvpr_chesapeake_landcover/train_text_data/train_data.json \
    --image_folder ./chesapeake_landcover/train/img \
    --gt_image_folder ./chesapeake_landcover/train/label \
    --mm_tunable_parts="mm_vis_encoder_decoder" \
    --mm_vision_tower_lr=1e-4 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --run_name "MID_RUN_NAME" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True\
    --load_InternImage_weigths False \
    --load_Other_weigths False \
    --output_dir "./checkpoints" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
