# AI Edge Examples

This module offers illustrations of how to utilize and run exported models. The examples provided are designed to be concise and have limited dependencies on third-party libraries. Our intention is for developers to leverage these examples as a starting point for integrating the exported models with their unique model-specific pipelines and requirements.

## Notes:

* If compiling the examples to run on an Android device, you need to download Android NDK and SDK and set `$ANDROID_NDK_HOME` and `$ANDROID_HOME` environment variables. Please note that _bazel_ currently only supports NDK versions 19, 20, and 21.

* Reducing memory: By default XNNPACK is enabled for optimized inference. XNNPACK needs to re-format the weights depending on the hardware of the device, and so has to create a new copy of the weights for the Dense layers. We can reduce our peak memory and remove the duplication of the weights by taking advantage of the XNNPACK weight caching mechanism. To enable this, add the flag `--weight_cache_path=/path/to/tmp/my_model.xnnpack_cache`. During the very first instance of the model, XNNPACK will build and write the cache into memory. Subsequent runs will load this cache which will also speed up initialization time. Currently it is the user's responsibility to ensure the validity of the cache (e.g. setting the proper path, removing the cache if needed, updating cache (by removing and rebuilding) if the model weights are updated, new/re-building cache for different models).

## Text Generation

In `text_generator_main.cc`, we provide an example of running a decoder-only model end-to-end using TensorFlow Lite as our inference engine.

To get started, you will need an exported  model with two signatures: `prefill` and `decode`. The example takes in an input prompt, tokenizes it, "prefills" the model with the tokens, and decodes autoregressively with greedy sampling until a stop condition is met. Finally, it detokenizes the generated token IDs into text.

It's important to note that while we use [SentencePiece](https://github.com/google/sentencepiece) as the tokenizer module in our example, it's not a requirement, and other tokenizers can be used as needed. Additionally, we're using a greedy sampling strategy, which simply takes an argmax over the output logits. There are many other options available that have been shown to generate better results.

As an example, you can run `text_generator_main`  for an exported Gemma model as follows:

```
bazel run -c opt //ai_edge_torch/generative/examples/c++:text_generator_main -- --tflite_model=PATH/gemma_it.tflite  --sentencepiece_model=PATH/tokenizer.model --start_token="<bos>" --stop_token="<eos>" --num_threads=16 --prompt="Write an email:" --weight_cache_path=PATH/gemma.xnnpack_cache
```

In `text_generator_main.cc`, pay close attention to how external KV Cache buffers are managed. These buffers are manually allocated and then propagated to the interpreter using `TfLiteCustomAllocation`.
This approach eliminates unnecessary data movements between calls, resulting in optimal overall performance.

It's important to note that not all delegates support this in-place update. For those cases, it's necessary to implement a ping-pong buffer and update the pointers between inference calls.
