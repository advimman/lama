#!/usr/bin/env python3

import os

import cv2
import numpy as np
import tqdm

from saicinpainting.evaluation.data import PrecomputedInpaintingResultsDataset
from saicinpainting.evaluation.utils import load_yaml


def main(args):
    config = load_yaml(args.config)

    if not args.predictdir.endswith('/'):
        args.predictdir += '/'

    dataset = PrecomputedInpaintingResultsDataset(args.datadir, args.predictdir, **config.dataset_kwargs)

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    for img_i in tqdm.trange(len(dataset)):
        try:
            pred_fname = dataset.pred_filenames[img_i]
            cur_out_fname = os.path.join(args.outpath, pred_fname[len(args.predictdir):])
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            sample = dataset[img_i]
            img = sample['image']
            mask = sample['mask']
            inpainted = sample['inpainted']

            cur_res = (1 - mask) * img + mask * inpainted

            cur_res = np.clip(np.transpose(cur_res, (1, 2, 0)) * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
        except Exception as ex:
            print(f'Failed to process {img_i}, {pred_fname} due to {ex}')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to evaluation config')
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('predictdir', type=str,
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
    aparser.add_argument('outpath', type=str, help='Where to put results')

    main(aparser.parse_args())
