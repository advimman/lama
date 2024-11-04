import logging
import os
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint, DefaultInpaintingTrainingModule

import numpy as np

np.bool = np.bool_

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

LOGGER = logging.getLogger(__name__)

model_dir = 'places_lama'  # or 'big-lama'

checkpoint_path = f'{model_dir}/models/best.ckpt'

train_config_path = os.path.join(model_dir, 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

print(f"train_config: {train_config}")

inpainting_model: DefaultInpaintingTrainingModule = load_checkpoint(train_config, checkpoint_path, strict=False,
                                                                    map_location='cpu')
inpainting_model.to('cpu')
print(f"model: {inpainting_model}")


class JITWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        batch = {
            "image": image,
            "mask": mask
        }
        out = self.model(batch)
        return out["inpainted"]


jit_model_wrapper = JITWrapper(inpainting_model)
jit_model_wrapper.to('cpu')

image = torch.rand(1, 3, 512, 512)
mask = torch.rand(1, 1, 512, 512)

traced_model = torch.jit.trace(jit_model_wrapper.eval(), (image, mask), strict=False).to("cpu")

traced_model.save(f'{model_dir}-jit.pt')
