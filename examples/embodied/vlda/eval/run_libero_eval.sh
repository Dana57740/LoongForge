#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Run the shared evaluator with a VLDA Stage 2 checkpoint.
# CHECKPOINT_PATH=/path/to/steps_80000 CUDA_VISIBLE_DEVICES=0 \
#   bash examples/embodied/vlda/eval/run_libero_eval.sh [libero_object ...]
# Or set CONFIG to an edited configs/libero/libero.yaml; all eval settings then come from YAML.
# Default: four suites, all tasks, 50 episodes/task.
# Smoke: MAX_TASKS=1 EPISODES_PER_TASK=1 and pass libero_object.
# SEED controls the client; server seeding is opt-in via --server-seed N.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SERVER_PYTHON=${SERVER_PYTHON:-$REPO_ROOT/.venv/bin/python}
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-$REPO_ROOT/.venv-libero/bin/python}
EVAL_DIR=${EVAL_DIR:-$REPO_ROOT/outputs/vlda/eval_$(date +%Y%m%d_%H%M%S)}

export LIBERO_ROOT=${LIBERO_ROOT:-$REPO_ROOT/third_party/LIBERO}
export PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}

if [[ -n ${CONFIG:-} ]]; then
    if [[ "$CONFIG" != /* ]]; then
        CONFIG="$REPO_ROOT/$CONFIG"
    fi
    exec "$BENCHMARK_PYTHON" -u -m loongforge.embodied.eval.orchestrator.run --config "$CONFIG"
fi

"$SERVER_PYTHON" "$SCRIPT_DIR/prepare_libero_config.py" \
    --checkpoint-path "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a LoongForge Stage 2 checkpoint directory}" \
    --output-dir "$EVAL_DIR" \
    --tokenizer-path "${TOKENIZER_PATH:-$REPO_ROOT/models/paligemma_tokenizer.model}" \
    --port "${PORT:-12093}" --health-port "${HEALTH_PORT:-12094}" \
    --seed "${SEED:-7}" --max-tasks "${MAX_TASKS:-0}" \
    --episodes-per-task "${EPISODES_PER_TASK:-50}" --max-steps "${MAX_STEPS:--1}" \
    "$@"

if [[ ${PREPARE_ONLY:-0} == 1 ]]; then
    exit 0
fi

"$BENCHMARK_PYTHON" -u -m loongforge.embodied.eval.orchestrator.run \
    --config "$EVAL_DIR/eval.yaml" 2>&1 | tee -a "$EVAL_DIR/client.log"
