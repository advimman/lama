# Example transformer models (decoder-only LLMs)
Here we provide a list of popular decoder-only LLMs composed via the transformer building blocks from this library. The main purpose is to demonstrate how to construct a new PyTorch LLM model from scratch using the AI Edge Torch Generative API, and convert it to TFLite format for on-device inference.

## Gemma
Gemma is Google's open-source LLM. The model has both a 2B and 7B versions. See the [model's Kaggle page](https://www.kaggle.com/models/google/gemma-2). The example we provide is Gemma 2B, and the checkpoint for the PyTorch model can be downloaded from [here](https://www.kaggle.com/models/google/gemma-2/pyTorch/gemma-2-2b-it).

## Llama
[Llama 3.2 model](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
is Meta's open-source LLM with 1B and 3B for text, and 11B and 90B for vision.
The examples we provide are Llama 3.2 1B and 3B for text. The checkpoint can be
found [here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main).

## TinyLlama
[TinyLlama](https://github.com/jzhang38/TinyLlama) is a popular OSS smaller version of Meta's Llama2 model, with only 1.1B parameters. [HuggingFace checkpoint](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).

## Microsoft Phi-2 and 3.5-mini
Microsoft Phi-2 and Phi-3.5-mini are also decoder-only LLMs with 2.7B and 3.82B
parameters each. See details on
[Kaggle](https://www.kaggle.com/models/Microsoft/phi/transformers/2) for Phi-2
and [HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) for
Phi-3.5-mini. Note that the example of Phi-3.5-mini supports up to 4K tokens,
not to 128K tokens which the original Phi-3.5 supports.

## Apple OpenELM
[Apple OpenELM](https://huggingface.co/apple/OpenELM) is also a decoder-only LLM
with 270M, 450M, 1.1B, and 3B parameters.  The example we provide is OpenELM 3B,
and the checkpoint for the model can be found
[here](https://huggingface.co/apple/OpenELM-3B/tree/main).

## HuggingFace SmolLM
[HuggingFace SmolLM](https://huggingface.co/blog/smollm) is also a decoder-only
LLM with 135M, 360M, 1.7B parameters. The example we provide is SmolLM 135M, and
the checkpoint for the model can be found
[here](https://huggingface.co/HuggingFaceTB/SmolLM-135M).

## Qwen
Alibaba's [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
0.5B, 1B, 3B modes are also provided as examples.

## AMD-Llama-135m

[AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m) is a 135M parameter model based on the Llama2 architecture and uses the same tokenizer as Llama2. It was trained on AMD Instinct MI250 accelerators. 

## Overall workflow
To support a new LLM with the Edge Generative API, we need to go through the process of model (re)authoring, checkpoint mapping/loading, model quantization (via PT2E), model conversion to flatbuffer schema, model quality evaluation, benchmarking and on-device inference pipeline authoring.

### Model (re)authoring
Model (re)authoring refers to the process of a few things:
1) Understanding the overall model architecture (encoder-decoder, decoder-only etc).
2) Compose the model using `ai_edge_torch` provided transformer building blocks.
For each of the example models, we have a model definition file (e.g. tiny_llama/tiny_llama.py) where a `nn.Module` is defined, with its layers and a forward function. There is also a `get_model_config` function which returns a `ModelConfig` instance with hyper-parameters such as embedding size, layer count etc. Finally, there is a `define_and_run` function which builds the model instance, and runs the forward pass with a few sample inputs.

Here we use `TinyLlama` as an example to walk you through the authoring steps.

#### Define model's structure
The model structure of `TinyLlama` can be found by instantiating the pretrained model.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(model)
```

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-21): 22 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
```

Based on the original model structure, construct a new nn.Module model using
the AI Edge Torch Generative API. As many examples do, either use
[`DecoderOnlyModel`](https://github.com/protobird-git/ai-edge-torch/blob/main/ai_edge_torch/generative/utilities/model_builder.py)
class as is like [SmolLM](https://github.com/protobird-git/ai-edge-torch/blob/main/ai_edge_torch/generative/examples/smollm/smollm.py),
or inherit [`DecoderOnlyModel`](https://github.com/protobird-git/ai-edge-torch/blob/main/ai_edge_torch/generative/utilities/model_builder.py)
class then modify only some component like
[Llama 3.2](https://github.com/protobird-git/ai-edge-torch/blob/main/ai_edge_torch/generative/examples/llama/llama.py),
or construct entirely a new nn.Module from scratch like
[Gemma 2](https://github.com/protobird-git/ai-edge-torch/blob/main/ai_edge_torch/generative/examples/gemma/gemma2.py).

Here is an example of TinyLlama constructed from scratch.

https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/examples/tiny_llama/tiny_llama.py#L46-L77

#### Define model's forward function
https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/examples/tiny_llama/tiny_llama.py#L79-L104

#### Set model's hyper parameters
It's nice to have the new nn.Module configurable.
https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/examples/tiny_llama/tiny_llama.py#L107-L132

These config values can be found in
[`config.json`](https://www.kaggle.com/datasets/ansonkw/tinyllama-1-1b-chat-v1-0?select=config.json)
of the Kaggle page.

Now, you will have an `nn.Module` named `TinyLlama`, the next step is to restore the weights from original checkpoint into the new model.

### Checkpoint mapping/loading
After the model is defined, we need to load the original trained weights to the
new model. This is needed because the `state_dict` of the new model will be
different from the original model's `state_dict`. There are helper functions in
place to simplify the `state_dict` mapping process (`utilities/loader.py`).
The user needs to provide a layer name template (TensorNames) for the source
model. For `TinyLlama`, layer names can be found from the SafeTensors file.

```python
import ai_edge_torch.generative.utilities.loader as loading_utils

safetensors = loading_utils.load_safetensors("path_to_checkpoint")
print(safetensors.keys())
```
```
dict_keys(['lm_head.weight', 'model.embed_tokens.weight', 'model.layers.0.input_layernorm.weight', 'model.layers.0.mlp.down_proj.weight', ...])
```

This template is then used to create an updated `state_dict` that works
with the mapped model. The tensor map includes the following fields:
https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/utilities/loader.py#L94-L109

The fields that have a default value of `None` are optional and should only be
populated if they are relevant to the model architecture. For `TinyLlama`, we
will define the following `TENSOR_NAMES`:
https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/examples/tiny_llama/tiny_llama.py#L30-L43

With the `TensorNames` defined, a user can simply use the loading utils to load
an instance of the mapped model. For instance:

```python
model = MappedModel(config)
loader = loading_utils.ModelLoader("path_to_checkpoint", TENSOR_NAMES)
loader.load(model)
```

Currently, `ModelLoader` supports PyTorch state dictionary and SafeTensors
checkpoints. We recommend testing the mapped model against your reference implementation
using a few input samples before proceeding to the conversion step.

### Verify (re)authored model
Once the model (re)authoring is completed, it should be verified if it generates
the output close to one from the original model. Generative API provides some
utilities to make it easy to verify models as shown with `verify.py` in each
example folder.

To instantiate the original models, `verify.py` imports `kagglehub` and/or
`transformers` which may require user authentication tokens to download the
original models. Please refer
[Kagglehub page](https://www.kaggle.com/docs/api#authentication) or
[HuggingFace page](https://huggingface.co/docs/hub/en/security-tokens)
about how to set user authentication tokens up.

To verify Gemma models, it requires to install `gemma_pytorch` package from its
github repository.

```bash
pip install -q -U immutabledict sentencepiece
git clone https://github.com/google/gemma_pytorch.git
export PYTHONPATH=$PWD/gemma_pytorch:$PYTHONPATH
```

### Model conversion
In this step, we use the `ai_edge_torch`'s standard multi-signature conversion API to convert PyTorch `nn.Module` to a single TFLite flatbuffer for on-device execution. For example, in `tiny_llama/convert_to_tflite.py`, we use this python code to convert the `TinyLlama` model to a multi-signature TFLite model:
https://github.com/google-ai-edge/ai-edge-torch/blob/853301630f2b2455bd2e2f73d8a47e1a1534c91c/ai_edge_torch/generative/examples/tiny_llama/convert_to_tflite.py#L26-L61

Once converted, you will get a `.tflite` model which will be ready for on-device execution. Note that the `.tflite` model generated uses static shapes. Inside the generated `.tflite` model, there will be two signatures defined (two entrypoints to the model):
1) `prefill`: taking 2 tensor inputs `prefill_tokens`, `prefill_input_pos`. With shape `(BATCH_SIZE, PREFILL_SEQ_LEN)` and `(PREFILL_SEQ_LEN)`.
2) `decode`: taking 2 tensor inputs `decode_token`, `decode_input_pos`. With shape `(1, 1)` and `(1)`.
To learn more about TFLite signatures, please refer to this [article](https://www.tensorflow.org/lite/guide/signatures).

### Model quantization
To apply quantization, we need to create a configuration that fully expresses how the model should be quantized. This configuration is then passed into conversion, generating a quantized model.

`quantize/quant_recipes.py` contains a list of recipes that are known to be well-supported during runtime. For the average user, this is a good starting point to select the quantization scheme that is best suited for your deployment needs. After identifying the target recipe, the model can be quantized as follows. This example is extracted from `generative/examples/quantize/example.py`.

```python
quant_config = quant_recipes.full_int8_dynamic_recipe()
edge_model = ai_edge_torch.convert(
    model, (tokens, input_pos), quant_config=quant_config
)
```
Once converted, you will get a quantized `.tflite` model which will be ready for on-device execution.
