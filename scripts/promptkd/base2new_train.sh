#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0
