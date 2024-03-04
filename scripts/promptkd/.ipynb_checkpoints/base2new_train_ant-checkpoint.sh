#!/bin/bash

# custom config
DATA='/ossfs/workspace/nas1/209290/zhengli_dataset'
# DATA='/root/'
TRAINER=PromptKD

DATASET=caltech101

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

for DATASET in 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
do
    for SEED in 1 2
    do
        DIR=/ossfs/workspace/nas1/209290/zhengli_exp_storage/paper_reproduce/${DATASET}/seed_${SEED}_reproduce
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
    done
done
