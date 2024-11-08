# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = [
        "//third_party/py/ai_edge_torch:license",
    ],
    default_visibility = [
        "//third_party/py/ai_edge_torch:__subpackages__",
    ],
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    deps = [
        ":model_config",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "blocks_2d",
    srcs = ["blocks_2d.py"],
    deps = [
        ":builder",
        ":model_config",
        "//third_party/py/ai_edge_torch/generative/layers:attention",
        "//third_party/py/ai_edge_torch/generative/layers:builder",
        "//third_party/py/ai_edge_torch/generative/layers:model_config",
        "//third_party/py/torch:pytorch",
    ],
)

py_library(
    name = "model_config",
    srcs = ["model_config.py"],
    deps = ["//third_party/py/ai_edge_torch/generative/layers:model_config"],
)
