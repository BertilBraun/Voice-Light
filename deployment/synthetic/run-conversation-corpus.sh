#!/usr/bin/env bash
set -euo pipefail

source /opt/supervisor-scripts/utils/environment.sh

repository="${VOICE_LIGHT_CORPUS_REPOSITORY:?VOICE_LIGHT_CORPUS_REPOSITORY is required}"
output="${VOICE_LIGHT_CORPUS_OUTPUT:?VOICE_LIGHT_CORPUS_OUTPUT is required}"
qwen_environment="${VOICE_LIGHT_QWEN_ENVIRONMENT:?VOICE_LIGHT_QWEN_ENVIRONMENT is required}"
vllm_environment="${VOICE_LIGHT_VLLM_ENVIRONMENT:?VOICE_LIGHT_VLLM_ENVIRONMENT is required}"
cosy_environment="${VOICE_LIGHT_COSY_ENVIRONMENT:?VOICE_LIGHT_COSY_ENVIRONMENT is required}"
cosy_repository="${VOICE_LIGHT_COSY_REPOSITORY:?VOICE_LIGHT_COSY_REPOSITORY is required}"
target_planned_hours="${VOICE_LIGHT_TARGET_PLANNED_HOURS:-27.0}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTHONUNBUFFERED=1
export UV_NO_CACHE=1

qwen_model_revision=5ecdb67327fd37bb2e042aab12ff7391903235d3
cosy_model_revision=29e01c4e8d000f4bcd70751be16fa94bf3d85a18
cosy_runtime_revision=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

run_prompt_stage() {
  local stage_output="$1"
  local set_id="$2"
  shift 2
  if [[ ! -f "$stage_output/prompts.json" ]]; then
    export PYTHONPATH="$repository"
    "$vllm_environment/bin/python" \
      -m app.local.synthetic_generation.generate_conversation_prompts_vllm \
      --set-id "$set_id" \
      --seed 260830 \
      --output "$stage_output/prompts.json" \
      --generation-batch-size 64 \
      "$@"
  fi
  export PYTHONPATH="$repository"
  "$qwen_environment/bin/python" \
    -m app.local.synthetic_generation.conversation_corpus_audit \
    --prompts "$stage_output/prompts.json" \
    --output "$stage_output/audit-prompts.json"
}

run_reference_stage() {
  local stage_output="$1"
  export PYTHONPATH="$repository"
  "$qwen_environment/bin/python" \
    -m app.local.synthetic_generation.generate_conversation_qwen_references \
    --prompts "$stage_output/prompts.json" \
    --output "$stage_output/references" \
    --model-revision "$qwen_model_revision" \
    --batch-size 2
}

run_render_stage() {
  local stage_output="$1"
  export PYTHONPATH="$cosy_repository:$cosy_repository/third_party/Matcha-TTS:$repository"
  "$cosy_environment/bin/python" \
    -m app.local.synthetic_generation.generate_conversation_cosyvoice \
    --prompts "$stage_output/prompts.json" \
    --references "$stage_output/references/voice-references.json" \
    --output "$stage_output/cosyvoice" \
    --model-directory "$cosy_repository/pretrained_models/Fun-CosyVoice3-0.5B" \
    --model-id FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --model-revision "$cosy_model_revision" \
    --runtime-revision "$cosy_runtime_revision" \
    --batch-size 1
}

run_compile_stage() {
  local stage_output="$1"
  export PYTHONPATH="$repository"
  if [[ ! -f "$stage_output/compiled/synthetic-corpus.json" ]]; then
    "$qwen_environment/bin/python" -m app.local.synthetic_generation.cli \
      compile-conversations \
      --prompt-set "$stage_output/prompts.json" \
      --tts-manifest "$stage_output/cosyvoice/render.json" \
      --output "$stage_output/compiled" \
      --split-seed voice-light-synthetic-conversations-20h-v3 \
      --crop-variants 1 \
      --assistant-only-fraction 0.0 \
      --user-only-fraction 0.0 \
      --event-light-fraction 0.0 \
      --assistant-duration-variation 0.1 \
      --allow-incomplete-sampling-controls
  fi
  "$qwen_environment/bin/python" \
    -m app.local.synthetic_generation.conversation_corpus_audit \
    --prompts "$stage_output/prompts.json" \
    --tts-manifest "$stage_output/cosyvoice/render.json" \
    --corpus-manifest "$stage_output/compiled/synthetic-corpus.json" \
    --output "$stage_output/audit-complete.json"
}

run_review_stage() {
  local stage_output="$1"
  export PYTHONPATH="$repository"
  "$qwen_environment/bin/python" \
    -m app.local.synthetic_generation.build_conversation_clone_review \
    --prompts "$stage_output/prompts.json" \
    --references "$stage_output/references/voice-references.json" \
    --renders "$stage_output/cosyvoice/render.json" \
    --corpus "$stage_output/compiled" \
    --output "$stage_output/review/index.html"
}

mkdir -p "$output"
cd "$repository"

preflight="$output/preflight"
run_prompt_stage "$preflight" voice_light_conversation_preflight_v3 --count 10
run_reference_stage "$preflight"
run_render_stage "$preflight"
run_compile_stage "$preflight"
run_review_stage "$preflight"
touch "$preflight/PREFLIGHT_COMPLETE"
echo "PREFLIGHT_COMPLETE"

corpus="$output/corpus"
run_prompt_stage \
  "$corpus" \
  voice_light_synthetic_conversations_20h_v3 \
  --count 2000 \
  --target-conversation-hours "$target_planned_hours"
run_reference_stage "$corpus"
run_render_stage "$corpus"
run_compile_stage "$corpus"
touch "$corpus/CORPUS_COMPLETE"
echo "CORPUS_COMPLETE"
