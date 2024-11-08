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

"""Builds a SentencePieceModel protobuf from a HuggingFace tokenizer.

If a SentencePieceModel protobuf file is already available, it copies the
SentencePieceModel protobuf file instead of building a new one.

If not, it tries to build a SentencePieceModel protobuf file from the tokenizer
config files.

Please note that the SentencePirceModel protobuf would not output the same token
IDs as the tokenizer for all input strings because the conversion relies on
heuristics. For example, SentencePiece model built from Llama3.2 tokenizer with
"decode" normalization has around 1% mismatch ratio. It's user's responsibility
to verify the quality of the built SentencePiece model.
"""

import logging
import random
from typing import List

from absl import app
from absl import flags
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_model
import transformers

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    None,
    "The path to the checkpoint where the tokenizer config files are.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "The path of the output SentencePieceModel protobuf file.",
)

_STRINGS_TO_VERIFY = flags.DEFINE_list(
    "strings_to_verify",
    [
        "Hello, world! How are you?",
        "Instruct: write a python program to add 2 and 3.",
    ],
    "The strings to verify the SentencePieceModel protobuf file.",
)

_NORMALIZE_TOKENS = flags.DEFINE_enum(
    "normalize_tokens",
    "decode",
    ["none", "gpt2", "decode"],
    "Normalize tokens of the original tokenizer to be compatible with "
    "SentencePiece model.\n"
    "  none:   do not normalize the tokens\n"
    "  gpt2:   apply gpt-2 unicode_to_byte conversion\n"
    "  decode: call tokenizer.decode([token id]) for each token",
)

_NUM_PAIRS_TO_VERIFY = flags.DEFINE_integer(
    "num_pairs_to_verify",
    1000,
    "The number of pairs to verify the SentencePieceModel protobuf file.",
)


def _bytes_to_unicode():
  """Returns list of utf-8 byte and a corresponding list of unicode strings.

  It's a copy of https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9.
  """
  bs = (
      list(range(ord("!"), ord("~") + 1))
      + list(range(ord("¡"), ord("¬") + 1))
      + list(range(ord("®"), ord("ÿ") + 1))
  )
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


# An inverse map of _bytes_to_unicode() to decode unicode tokens in a HF
# transformers tokenizer into utf-8 tokens in the SentencePiece model.
_BYTE_DECODE_MAP = {v: k for k, v in _bytes_to_unicode().items()}


def _normalize_gpt2(token: str) -> str:
  """Normalizes a unicode character to a utf-8 character.

  It's a semantic copy of
  https://github.com/openai/gpt-2/blob/master/src/encoder.py#L105.
  """
  return bytearray(
      [_BYTE_DECODE_MAP[c] if c in _BYTE_DECODE_MAP else ord(c) for c in token]
  ).decode("utf-8", "replace")


_NORMALIZE_FUNCS = {
    "none": lambda x, id, _: x,
    "gpt2": lambda x, id, _: _normalize_gpt2(x),
    "decode": lambda _, id, tokenizer: tokenizer.decode([id]),
}


def _add_token(
    token: str,
    id: int,
    tokenizer: transformers.PreTrainedTokenizer,
    sp_model: spm_model.ModelProto,
    tokens_seen: set[str],
    counts: dict[spm_model.ModelProto.SentencePiece.Type, int],
):
  """Adds a token to the SentencePieceModel protobuf with a derived type."""
  unk_token = tokenizer.unk_token or tokenizer.pad_token or tokenizer.eos_token
  if token == unk_token:
    type = spm_model.ModelProto.SentencePiece.UNKNOWN
  elif token in tokenizer.special_tokens_map:
    type = spm_model.ModelProto.SentencePiece.CONTROL
    sp_model.trainer_spec.control_symbols.append(token)
  elif token in tokenizer.get_added_vocab():
    type = spm_model.ModelProto.SentencePiece.USER_DEFINED
    sp_model.trainer_spec.user_defined_symbols.append(token)
  else:
    type = spm_model.ModelProto.SentencePiece.NORMAL

  count_type = type
  normalized = _NORMALIZE_FUNCS[_NORMALIZE_TOKENS.value](token, id, tokenizer)
  if normalized == token:
    pass
  elif normalized in tokens_seen:
    logging.debug(
        'DUPLICATE: token "%s"(id=%d) normalized to "%s"', token, id, normalized
    )
    normalized = token
    # Change only the type of counts for logging. When UNUSED is set for SPM
    # model, it seems to have some negative impact, i.e. the ratio of mismatched
    # ID pairs is slightly higher.
    count_type = spm_model.ModelProto.SentencePiece.Type.UNUSED
  else:
    tokens_seen.add(normalized)
  sp_model.pieces.add(piece=normalized, score=-id, type=type)
  counts[count_type] = counts.get(count_type, 0) + 1


def _build_spm_model_from_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
) -> spm_model.ModelProto:
  """Builds a SentencePieceModel protobuf from a tokenizer."""
  sp_model = spm_model.ModelProto()
  sp_model.trainer_spec.model_type = spm_model.TrainerSpec.BPE
  sp_model.trainer_spec.vocab_size = len(tokenizer.vocab)
  sp_model.normalizer_spec.add_dummy_prefix = False
  sp_model.normalizer_spec.remove_extra_whitespaces = False
  sp_model.normalizer_spec.escape_whitespaces = False
  sp_model.denormalizer_spec.CopyFrom(sp_model.normalizer_spec)

  id_to_token = {id: tk for tk, id in tokenizer.vocab.items()}
  tokens_seen = set(tokenizer.vocab.keys())
  counts = {}
  for id in range(len(tokenizer.vocab)):
    _add_token(id_to_token[id], id, tokenizer, sp_model, tokens_seen, counts)

  logging.info("number of tokens: %d", len(sp_model.pieces))
  for type in counts:
    logging.info(
        "number of %s: %d",
        spm_model.ModelProto.SentencePiece.Type.Name(type),
        counts[type],
    )

  return sp_model


def _is_same_ids(ids_by_tokenizer: List[int], ids: List[int]) -> bool:
  """Checks if the IDs are the same to ones by transformer tokenizer."""
  # Transformer tokenizer may insert BOS token at the beginning.
  return ids_by_tokenizer == ids or ids_by_tokenizer[1:] == ids


def _log_not_matched(
    num_not_matched_strict: int, num_not_matched_loose: int, total: int
):
  """Logs the number of not matched pairs."""
  logging.info(
      "Not matched strictly %d/%d pairs: %.2f%%, loosely %d/%d pairs: %.2f%%",
      num_not_matched_strict,
      total,
      100 * num_not_matched_strict / total,
      num_not_matched_loose,
      total,
      100 * num_not_matched_loose / total,
  )


def _verify_spm_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
    spm_tokenizer: spm.SentencePieceProcessor,
):
  """Verifies the SentencePiece tokenizer."""
  # First, check if the token IDs encoded by the original tokenizer are the same
  # as the token IDs encoded by the SentencePiece tokenizer.
  for string in _STRINGS_TO_VERIFY.value:
    ids_by_tokenizer = tokenizer.encode(string)
    ids_by_spm = spm_tokenizer.encode(string)
    logging.info("String to verify: %s", string)
    logging.info("Token IDs by the oringal tokenizer: %s", ids_by_tokenizer)
    logging.info("Token IDs by the SentencePiece tokenizer: %s", ids_by_spm)
    if _is_same_ids(ids_by_tokenizer, ids_by_spm):
      logging.info("PASS")
    else:
      logging.warning("FAIL")

  # Second, check if how many strings decoded from the pairs of tokens by the
  # original tokenizer are encoded to the same token IDs by the SentencePiece
  # tokenizer.
  total = _NUM_PAIRS_TO_VERIFY.value
  num_not_matched_strict = 0
  num_not_matched_loose = 0
  for i in range(total):
    id_pair = random.sample(list(range(len(tokenizer.vocab))), 2)
    string = tokenizer.decode(id_pair)
    ids_by_tokenizer = tokenizer.encode(string)
    ids_by_spm = spm_tokenizer.encode(string)
    if not _is_same_ids(ids_by_tokenizer, ids_by_spm):
      num_not_matched_strict += 1
      if _is_same_ids(ids_by_tokenizer, id_pair):
        num_not_matched_loose += 1
        logging.debug(
            'NOT MATCHED: "%s", ids=%s, tok=%s, spm=%s',
            string,
            id_pair,
            ids_by_tokenizer,
            ids_by_spm,
        )
    if (i + 1) % 100 == 0:
      _log_not_matched(num_not_matched_strict, num_not_matched_loose, i + 1)
  _log_not_matched(num_not_matched_strict, num_not_matched_loose, total)


def main(_):
  tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT.value)
  if hasattr(tokenizer, "vocab_file"):
    logging.info("vocab_file exists: %s", tokenizer.vocab_file)
    with open(tokenizer.vocab_file, "rb") as f:
      sp_model = spm_model.ModelProto.FromString(f.read())
  else:
    logging.info("vocab_file does not exist. Try to build a new one.")
    sp_model = _build_spm_model_from_tokenizer(tokenizer)

  spm_serialized = sp_model.SerializeToString()
  spm_tokenizer = spm.SentencePieceProcessor()
  spm_tokenizer.LoadFromSerializedProto(spm_serialized)
  _verify_spm_tokenizer(tokenizer, spm_tokenizer)

  logging.info(
      "Writing the SentencePieceModel protobuf file to: %s", _OUTPUT_PATH.value
  )
  with open(_OUTPUT_PATH.value, "wb") as f:
    f.write(spm_serialized)


if __name__ == "__main__":
  app.run(main)
