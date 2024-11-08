/* Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/util.h"

// This is a simplified example of using TFLite to generate text.
// Please note that this is only a starting point and the user is expected
// to create their own pipeline potentially using different tokenizers and
// better samplers.
//
// Example usage:
//  generate_main --tflite_model="PATH/model.tflite" \
//    --sentencepiece_model="PATH/sp.model" \
//    --prompt="Write an email:" \
//    --max_decode_steps=64 \
//    --start_token="<bos>" \
//    --stop_token="<eos>"  \
//    --num_threads=4

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model prepared for text generation using ODML "
          "tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to sentencepiece model.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt to the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "The number of tokens to generate. Defaults to maximum Sequence size "
          "defined during conversion.");
ABSL_FLAG(std::string, start_token, "",
          "Start token is appended to the beginning of input prompt to "
          "signify start of sentence.");
ABSL_FLAG(std::string, stop_token, "",
          "Stop token used to deterine end of decoding loop. If not provided "
          "will decode until max_Seq_len or max_decode_steps.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "XNNPACK weight caching path, e.g. /tmp/model.xnnpack_cache.");

namespace {

// Prepare helpers
std::unique_ptr<tflite::FlatBufferModel> LoadModel() {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          absl::GetFlag(FLAGS_tflite_model).c_str());
  TFLITE_MINIMAL_CHECK(model != nullptr);
  return model;
}

void ApplyXNNPACKWeightCaching(tflite::Interpreter* interpreter) {
  auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
  delegate_options.weight_cache_file_path = weight_cache_path.c_str();
  delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

  TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                           tflite::Interpreter::TfLiteDelegatePtr(
                               TfLiteXNNPackDelegateCreate(&delegate_options),
                               [](TfLiteDelegate* delegate) {
                                 TfLiteXNNPackDelegateDelete(delegate);
                               })) == kTfLiteOk);
}

std::unique_ptr<tflite::Interpreter> BuildInterpreter(
    tflite::FlatBufferModel* model, int num_threads) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  TFLITE_MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  if (!absl::GetFlag(FLAGS_weight_cache_path).empty()) {
    // optionally use xnnpack with weight caching
    ApplyXNNPACKWeightCaching(interpreter.get());
  }

  return interpreter;
}

// TF Lite requires all buffers (including external buffers used for KV cache
// here) be `tflite::kDefaultTensorAlignment` aligned. To ensure that, we use
// this custom allocator. Please use with caution as different platforms may
// have different alignment requirements.
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  T* allocate(std::size_t n) {
    void* ptr;
    std::size_t size = n * sizeof(T);
    std::size_t padding = tflite::kDefaultTensorAlignment -
                          (size % tflite::kDefaultTensorAlignment);
    size += padding;
    int ret = posix_memalign(&ptr, tflite::kDefaultTensorAlignment, size);
    if (ret != 0) {
      return nullptr;
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t n) { free(p); }
};

std::map<std::string, std::vector<float, AlignedAllocator<float>>> BuildKVCache(
    tflite::Interpreter* interpreter) {
  tflite::SignatureRunner* runner = interpreter->GetSignatureRunner("prefill");
  if (runner == nullptr) {
    return {};
  }
  // The two arguments excluded are `tokens` and `input_pos`.
  size_t num_layers = (runner->input_size() - 2) / 2;
  if (num_layers == 0) {
    return {};
  }

  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
  for (int i = 0; i < num_layers; ++i) {
    std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
    std::string v_cache_name = "kv_cache_v_" + std::to_string(i);
    // We are assuming K and V tensors are of the same shape.
    TfLiteTensor* tensor = runner->input_tensor(k_cache_name.c_str());
    size_t count = tensor->bytes / sizeof(float);
    kv_cache.emplace(k_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
    kv_cache.emplace(v_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
  }

  return kv_cache;
}

tflite::SignatureRunner* GetSignatureRunner(
    tflite::Interpreter* interpreter, const std::string& signature_name,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache) {
  tflite::SignatureRunner* runner =
      interpreter->GetSignatureRunner(signature_name.c_str());
  for (auto& [name, cache] : kv_cache) {
    TfLiteCustomAllocation allocation = {
        .data = static_cast<void*>(cache.data()),
        .bytes = cache.size() * sizeof(float)};
    // Both input and output tensors are set to the same buffer. Not all
    // delegates support this in-place update. For those cases, we need to do
    // a ping-pong buffer and update the pointers between inference calls.
    TFLITE_MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(
                             name.c_str(), allocation) == kTfLiteOk);
    TFLITE_MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(
                             name.c_str(), allocation) == kTfLiteOk);
  }
  TFLITE_MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
  return runner;
}

std::unique_ptr<sentencepiece::SentencePieceProcessor>
LoadSentencePieceProcessor() {
  std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model),
                      std::ios::binary);
  std::string serialized_proto = std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  TFLITE_MINIMAL_CHECK(
      processor->LoadFromSerializedProto(serialized_proto).ok());
  return processor;
}

// A basic greedy sampler (equivalent to argmax).
int GreedySampler(const TfLiteTensor* logits) {
  float max_value = -std::numeric_limits<float>::infinity();
  int max_index = 0;
  // logits shape: [Batch, Seq, Vocab], Dtype: float
  for (int i = 0; i < logits->dims->data[2]; ++i) {
    if (logits->data.f[i] > max_value) {
      max_value = logits->data.f[i];
      max_index = i;
    }
  }
  return max_index;
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Prepare required components.
  std::unique_ptr<tflite::FlatBufferModel> model = LoadModel();
  std::unique_ptr<tflite::Interpreter> interpreter =
      BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor =
      LoadSentencePieceProcessor();
  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache =
      BuildKVCache(interpreter.get());
  TFLITE_MINIMAL_CHECK(!kv_cache.empty())

  // Get prefill and decode signature runners and allocate tensors per
  // signature.
  tflite::SignatureRunner* prefill_runner =
      GetSignatureRunner(interpreter.get(), "prefill", kv_cache);
  TFLITE_MINIMAL_CHECK(prefill_runner != nullptr);
  tflite::SignatureRunner* decode_runner =
      GetSignatureRunner(interpreter.get(), "decode", kv_cache);
  TFLITE_MINIMAL_CHECK(decode_runner != nullptr);

  // Get Input Tensors for each of the runners.
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* prefill_input = prefill_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("input_pos");
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* decode_input = decode_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* decode_input_pos = decode_runner->input_tensor("input_pos");
  int max_seq_size = prefill_input->dims->data[1];

  // Tokenize the input prompt.
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  std::vector<int> prompt_tokens;
  TFLITE_MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

  std::string start_token = absl::GetFlag(FLAGS_start_token);
  if (!start_token.empty()) {
    prompt_tokens.insert(prompt_tokens.begin(),
                         sp_processor->PieceToId((start_token)));
  }
  std::string stop_token = absl::GetFlag(FLAGS_stop_token);
  int stop_token_id = -1;
  if (!stop_token.empty()) {
    stop_token_id = sp_processor->PieceToId((stop_token));
  }

  // Fill in the inputs (assuming one batch).
  // NOTE: We skip the last token and use that during decode.
  int prefill_seq_size =
      std::min(static_cast<int>(prompt_tokens.size()), max_seq_size);
  for (int i = 0; i < prefill_seq_size - 1; ++i) {
    prefill_input->data.i32[i] = prompt_tokens[i];
    prefill_input_pos->data.i32[i] = i;
  }
  TFLITE_MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);

  // Decode until max sequence size or user defined step limit, whichever is
  // smaller.
  // NOTE: max kv-cache size is *not* necessarily the same size as the max
  // sequence length. KV Cache buffer wraps around if exahusted before max
  // sequence length or stopping criteria reach.
  int max_decode_steps = absl::GetFlag(FLAGS_max_decode_steps) == -1
                             ? max_seq_size
                             : absl::GetFlag(FLAGS_max_decode_steps);
  int decode_steps =
      std::min(max_decode_steps, max_seq_size - prefill_seq_size);
  TFLITE_MINIMAL_CHECK(decode_steps > 0);

  std::vector<int> output_tokens;
  output_tokens.reserve(decode_steps);
  int next_token = prompt_tokens[prefill_seq_size - 1];
  int next_position = prefill_seq_size - 1;
  for (int i = 0; i < decode_steps; ++i) {
    decode_input->data.i32[0] = next_token;
    decode_input_pos->data.i32[0] = next_position;
    TFLITE_MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
    next_token = GreedySampler(decode_runner->output_tensor("logits"));
    output_tokens.push_back(next_token);
    next_position += 1;
    if (next_token == stop_token_id) {
      break;
    }
  }

  // Detokenize the generated output.
  std::string output_text;
  TFLITE_MINIMAL_CHECK(sp_processor->Decode(output_tokens, &output_text).ok());

  printf("Prompt:\n%s\nOutput text:\n%s\n", prompt.c_str(),
         output_text.c_str());

  return 0;
}
