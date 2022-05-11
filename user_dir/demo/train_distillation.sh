#!/bin/sh

DICT_DIM=1

TOTAL_NUM_UPDATES=1000000 # just run 1000k steps...
WARMUP_UPDATES=60000      # 6 percent of the number of update
LR=5e-04                  # Peak LR for polynomial LR scheduler.
HEAD_NAME=imdb_head       # Custom name for the classification head.
NUM_CLASSES=4             # Number of classes for the classification task.
MAX_SENTENCES=2048        # Batch size.
DATA_PATH=/path/to/your/data
OUT_DIR=/path/to/save/model/checkpoints

cd ../..

fairseq-train $DATA_PATH \
    --valid-subset valid,valt \
    --dataset-impl mmap \
    --save-dir $OUT_DIR \
    --max-positions 2048 \
    --batch-size $MAX_SENTENCES \
    --user-dir user_dir \
    --task feature_based_sentence_prediction \
    --distillation-target \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch feature_based_linear_model \
    --criterion distillation \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --weight-decay 0.01 --optimizer mixed_adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-update $TOTAL_NUM_UPDATES \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --find-unused-parameters \
    --update-freq 1 \
    --encoder-embed-dim 1000 \
    --feature-dropout 0.1 --pooler-dropout 0.1 \
    --distillation-alpha 1.0 \
    --num-workers 20 \
    --validate-interval-updates 10000 \
    --save-interval-updates 100000;
