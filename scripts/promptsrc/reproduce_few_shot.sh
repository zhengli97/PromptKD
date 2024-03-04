#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=PromptSRC

DATASET=$1
SHOTS=$2
WEIGHTSPATH=$3

CFG=vit_b16_c2_ep50_batch4_4+4ctx_few_shot
LOADEP=50

for SEED in 1 2 3
do
    MODEL_DIR=${WEIGHTSPATH}/${SHOTS}shot/seed${SEED}
    DIR=output/few_shot/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo " The results exist at ${DIR}"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done