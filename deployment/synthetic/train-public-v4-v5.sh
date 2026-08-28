#!/usr/bin/env bash
set -euo pipefail

repository="${VOICE_LIGHT_REPOSITORY:-/workspace/Voice-Light-conversation-corpus-v1}"
python_environment="${VOICE_LIGHT_TRAINING_ENVIRONMENT:-/workspace/tts-venvs/training}"
corpus_root="${VOICE_LIGHT_SYNTHETIC_CORPUS_ROOT:-/workspace/voice-light-public-v4-v5-dynamic-e08a194}"
output_root="${VOICE_LIGHT_TRAINING_OUTPUT:-/workspace/voice-light-synthetic-v4-v5-training-e08a194}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
cd "$repository"
mkdir -p "$output_root"

exec "$python_environment/bin/python" -m app.training.turn_taking.cli \
  "$output_root/adapter.pt" \
  --dynamic-synthetic-corpus "$corpus_root/v4" \
  --dynamic-synthetic-corpus "$corpus_root/v5" \
  --validation-hub-repository BertilBraun/voice-light-audio \
  --validation-hub-revision 56e68eb8fb1d42159483612f508b9ce27672f724 \
  --hub-cache-directory /workspace/.hf_training_models \
  --primary-objective turn_completion \
  --max-steps 3500 \
  --validation-interval-steps 125 \
  --minimum-steps-before-stopping 1000 \
  --batch-size 2 \
  --validation-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --data-loader-workers 4 \
  --precision bfloat16 \
  --augmentation-profile expanded
