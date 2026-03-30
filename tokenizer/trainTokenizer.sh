#!/usr/bin/env bash
set -euo pipefail

CORPUS="${CORPUS:-input/dataset.txt}"
OUT_DIR="${OUT_DIR:-tokenizer}"
MODEL_PREFIX="${MODEL_PREFIX:-andric_sp}"

VOCAB_SIZE="${VOCAB_SIZE:-8000}"
MODEL_TYPE="${MODEL_TYPE:-unigram}"
INPUT_SENTENCE_SIZE="${INPUT_SENTENCE_SIZE:-1000000}"
NUM_THREADS="${NUM_THREADS:-4}"
MAX_SENTENCE_LENGTH="${MAX_SENTENCE_LENGTH:-4096}"
SEED_SENTENCEPIECE_SIZE="${SEED_SENTENCEPIECE_SIZE:-200000}"
CHARACTER_COVERAGE="${CHARACTER_COVERAGE:-0.9995}"
SHUFFLE_INPUT_SENTENCE="${SHUFFLE_INPUT_SENTENCE:-true}"

mkdir -p "$OUT_DIR"

INPUT_FOR_TRAIN="$CORPUS"

EXTRA_ARGS=()
if spm_train --help 2>&1 | grep -q -- "--train_extremely_large_corpus"; then
  EXTRA_ARGS+=(--train_extremely_large_corpus=true)
fi


spm_train \
  --input="$INPUT_FOR_TRAIN" \
  --model_prefix="$OUT_DIR/$MODEL_PREFIX" \
  --vocab_size="$VOCAB_SIZE" \
  --model_type="$MODEL_TYPE" \
  --input_sentence_size="$INPUT_SENTENCE_SIZE" \
  --num_threads="$NUM_THREADS" \
  --max_sentence_length="$MAX_SENTENCE_LENGTH" \
  --seed_sentencepiece_size="$SEED_SENTENCEPIECE_SIZE" \
  --character_coverage="$CHARACTER_COVERAGE" \
  --shuffle_input_sentence="$SHUFFLE_INPUT_SENTENCE" \
  --normalization_rule_name=nmt_nfkc \
  --split_digits=true \
  --byte_fallback=false \
  "${EXTRA_ARGS[@]}"
