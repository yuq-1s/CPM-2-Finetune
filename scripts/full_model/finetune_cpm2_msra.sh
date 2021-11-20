#! /bin/bash

# WORKING_DIR=/mnt
WORKING_DIR=.

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8

HOST_FILE="${WORKING_DIR}/configs/host_files/hostfile-cpm2"

MP_SIZE=4

DATA_EXT=".json"
DATA_PATH="${WORKING_DIR}/data/nonexist"

LR=${1-0.000005}
GRAD_ACC=${2-4}

CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config.json"
CKPT_PATH="${WORKING_DIR}/cpm2_MP${MP_SIZE}"

SAVE_PATH="${WORKING_DIR}/results/msra/cpm2_finetune_lr${LR}const_G${GRAD_ACC}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

BATCH_SIZE=16
TRAIN_ITER=-1
EPOCHS=20

SEQ_LENGTH=128

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name msra"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 100000"
OPTS+=" --eval-interval 50"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
# OPTS+=" --do_eval"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --seq-length ${SEQ_LENGTH}"
OPTS+=" --out-seq-length ${SEQ_LENGTH}"
OPTS+=" --enc-seq-length ${SEQ_LENGTH}"
OPTS+=" --dec-seq-length ${SEQ_LENGTH}"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100 

# CMD="deepspeed --include localhost:0,1,2,3,4,5,6,7 ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"
# CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"
CMD="deepspeed --hostfile ${HOST_FILE} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log

