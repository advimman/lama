#!/bin/bash

DIRNAME="$(dirname $0)"
DATA_ROOT=/data/inpainting/Places365

for config in random_medium_256 random_medium_512 random_thick_256 random_thick_512 random_thin_256 random_thin_512
do
    MASKS_OUTPATH=$DATA_ROOT/val_large_random_masks/$config

    $DIRNAME/gen_mask_dataset.py $DIRNAME/../configs/data_gen/$config.yaml \
        $DATA_ROOT/val_large \
        $MASKS_OUTPATH \
        --n-jobs 4

    $DIRNAME/calc_dataset_stats.py $MASKS_OUTPATH \
        ${MASKS_OUTPATH}_summary \
        --samples-n 20
done

