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

import functools
import json
import os
from typing import List, Tuple
import unicodedata

import regex as re


def create_bytes_table() -> dict:
  table = {}
  special_count = 0
  for byte in range(256):
    category = unicodedata.category(chr(byte))
    if category[0] not in [
        'C',
        'Z',
    ]:  # ith character is NOT control char or space
      table[byte] = chr(byte)
    else:  # ith character IS control char or space
      table[byte] = chr(special_count + 256)
      special_count += 1
  return table


def pairwise(seq):
  a = iter(seq)
  b = iter(seq)
  next(b)
  return zip(a, b)


class Tokenizer:

  def __init__(self, vocab_dir: str):
    with open(os.path.join(vocab_dir, 'vocab.json'), encoding='utf-8') as f:
      self.vocab = json.load(f)

    with open(os.path.join(vocab_dir, 'merges.txt'), encoding='utf-8') as f:
      lines = f.read().split('\n')
      lines = lines[1:-1]
      self.merges = {tuple(bigram.split()): i for i, bigram in enumerate(lines)}

    self.bos_token = self.vocab['<|startoftext|>']
    self.eos_token = self.vocab['<|endoftext|>']
    self.pad_token = self.vocab['<|endoftext|>']
    self.max_length = 77
    self.bytes_table = create_bytes_table()
    self.chunk_pattern = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re.IGNORECASE,
    )

  def encode(self, text: str) -> List[int]:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()

    tokens = [self.bos_token]
    for chunk in re.findall(self.chunk_pattern, text):
      chunk = ''.join(self.bytes_table[byte] for byte in chunk.encode('utf-8'))
      tokens.extend(self.vocab[word] for word in self.bpe(chunk))
    tokens.append(self.eos_token)

    tokens = tokens[: self.max_length]
    token_length = len(tokens)
    pad_length = self.max_length - token_length
    tokens += [self.pad_token] * pad_length
    return tokens

  def encode_batch(self, texts: List[str]) -> List[List[int]]:
    return [self.encode(text) for text in texts]

  @functools.lru_cache(maxsize=10000)
  def bpe(self, chunk: str) -> Tuple[str]:
    words = list(chunk)
    words[-1] += '</w>'

    while len(words) > 1:
      valid_pairs = [pair for pair in pairwise(words) if pair in self.merges]
      if not valid_pairs:
        break

      bigram = min(valid_pairs, key=lambda pair: self.merges[pair])
      first, second = bigram

      new_words = []
      for word in words:
        if word == second and new_words and new_words[-1] == first:
          new_words[-1] = first + second
        else:
          new_words.append(word)
      words = new_words

    return tuple(words)
