# ODML Transformer Layers
Common Pytorch building blocks to re-author transformer models.

## Attention layers
`attention.py` and `attention_utils.py` contain common building blocks for the attention calculation, which is the key part of transformer models. You can use the abstractions provided here to compose your transformer model.

These two files provide the following common Python helper functions: 
* `scaled_dot_product_attention`: helper function to compute scaled dot product attention on query, key and value tensors.
* `scaled_dot_product_attention_with_hlfb`: same as `scaled_dot_product_attention` with the addition of HLFB (high-level function boundary) for improved performance.
* `build_rope_cache`: pre-compute sin and cos values for Rotary Positional Embedding.
* `build_causal_mask_cache`: build a cache for causal self attention mask.

And also the following `nn.Module` classes:
* `TransformerBlock`
* `CausalSelfAttention`
* `SelfAttention`
* `CrossAttention`

## Builder class for common layers
In `builder.py`, it provides following helper functions:
* `build_norm`: constructs different kinds of normalizers based on a config.
* `build_ff`: constructs different kinds of feed forward layers, which includes Sequential or Gated.

## Feed forward layer
The library provides the following `nn.Modules` to represent feed forward layer.
* `SequentialFeedForward`
* `GatedFeedForward`

## KV cache layer
We provide a `nn.Module` KVCache to express the logic to update the cache. It also has internal logic to apply HLFB to ensure high-performance at runtime.

## Model Configuration class
Currently, the library provides the following configuration class for you to customize the transformer model:
* `AttentionConfig`
* `FeedForwardConfig`
* `NormalizationConfig`
* `ModelConfig`

## Normalization layer
`normalization.py` provides normalization modules currently not supported by Pytorch such as `RMSNorm`:
* `RMSNorm`

## RoPE Embedding
`rotary_position_embedding.py` contains helper functions for applying RoPE to tensors.

## High-Level function boundary for performance
We introduce High-Level Function Boundary (HLFB) as a way of annotating performance-critical pieces of the model (e.g. `scaled_dot_product_attention`, or `KVCache`). HLFB allows the converter to lower the annotated blocks to performant TFLite custom ops. Following is an example of applying HLFB to `SDPA`:
https://github.com/google-ai-edge/ai-edge-torch/blob/25c764ad21e6f1fda5600dfc27406ef0424c8c3a/ai_edge_torch/generative/layers/scaled_dot_product_attention.py#L69-L117
