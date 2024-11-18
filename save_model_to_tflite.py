import torch
import numpy as np

np.bool = np.bool_
import ai_edge_torch
import os
import yaml
from omegaconf import OmegaConf
import logging
from saicinpainting.training.trainers import load_checkpoint, DefaultInpaintingTrainingModule

os.environ['PJRT_DEVICE'] = 'CPU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

LOGGER = logging.getLogger(__name__)

print('load model')

model_dir = 'places_lama'  # 'big-lama'
# model_dir = 'big-lama'  # 'big-lama'

checkpoint_path = f'{model_dir}/models/best.ckpt'
train_config_path = os.path.join(model_dir, 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

print(f'train_config: {train_config}')

model: DefaultInpaintingTrainingModule = load_checkpoint(train_config, checkpoint_path, strict=False,
                                                         map_location='cpu')
model.to('cpu')
print(model.eval())
model.freeze()

image = torch.rand(1, 512, 512, 3)
mask = torch.rand(1, 512, 512, 1)
sample_args = (image, mask,)

from torch import nn


class JITWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        image = image.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        masked_img = image * (1 - mask)
        input_t = torch.cat([masked_img, mask], dim=1)

        out = self.model(input_t)
        out = mask * out + masked_img
        out = out.permute(0, 2, 3, 1)
        return out


jit_model_wrapper = JITWrapper(model.generator)

print('ai_edge_torch.convert...')
edge_model = ai_edge_torch.convert(jit_model_wrapper.eval(), sample_args)
print('ai_edge_model.export...')
edge_model.export(f'{model_dir}.tflite')

# trans to fp16
# import ai_edge_torch.lowertools.torch_xla_utils as uu
# from ai_edge_torch import model
#
# dir = 'big_lama_saved_model'
#
# result = model.TfLiteModel(uu.custom2tflite(dir))
# result.export('big-lama-fp16-custom.tflite')