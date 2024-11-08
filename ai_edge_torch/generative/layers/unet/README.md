# ODML UNet Layers
Common PyTorch building blocks to re-author UNet based models.

## Blocks 2D layers
`blocks_2d.py` provides common building blocks used in AutoEncoder and general UNet-like models. Each block has a corresponding config class provided in `model_config.py`, and the block layer is initialized with the config class. `blocks_2d.py` provide the following blocks:
* `ResidualBlock2D`: a basic residual layer containing two convolution layers, with optional time embedding layer.
* `AttentionBlock2D`: self attention layer for 2D tensor.
* `CrossAttentionBlock2D`: cross attention layer for 2D tensor, between latent tensor and context tensor.
* `FeedForwardBlock2D`: basic feed forward layer used in transformer 2D block.
* `TransformerBlock2D`: building block for text-to-image diffusion models, containing `AttentionBlock2D`, `CrossAttentionBlock2D` and `FeedForwardBlock2D`.
* `DownEncoderBlock2D`: encoder block used in AutoEncoder and UNet, with optional down sampling layer.
* `UpDecoderBlock2D`: decoder block used in AutoEncoder and UNet, with optional up sampling layer.
* `SkipUpDecoderBlock2D`: decoder block used in UNet, with skip connections from encoder.
* `MidBlock2D`: middle block used in AutoEncoder and UNet.

## Builder class for common layers:
In `builder.py`, it provides following helper functions:
* `build_upsampling`
* `build_downsampling`

## Model config class
`model_config.py` provide the configs classes used in 2D blocks, utility layers and whole AutoEncoder and UNet model.
