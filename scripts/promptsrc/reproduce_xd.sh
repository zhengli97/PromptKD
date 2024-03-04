#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=PromptSRC

DATASET=$1
SEED=$2
WEIGHTSPATH=$3

CFG=vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets
SHOTS=16
LOADEP=20

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on evaluation datasets
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
    DATASET.NUM_SHOTS ${SHOTS} \

fi