# VLDA on LoongForge

VLDA is registered as its own model and dataset strategy. It uses the common
LoongForge training loop with a small trainer extension for V-JEPA2 teacher EMA
and checkpoint assets. The current target is LIBERO training with state tokens,
LoongForge normalization, and resume within this implementation.

## Environment and inputs

Use the embodied environment from `docker/Dockerfile`: LeRobot 0.5.0,
Transformers 5.3.0, and the project's PyTorch build. LeRobot 0.5.0 requires
Python 3.12. `sentencepiece`, `timm`, `pandas`, `pyarrow`, and `torchdata` are
also needed. The encoder-only V-JEPA2 sources are included with their MIT
license; an installed VLDA/openpi checkout or patched Transformers is not needed.

For the local VLDA environment, use Python 3.12, PyTorch `2.8.0+cu128`,
torchvision `0.23.0+cu128`, LeRobot `0.5.0`, Transformers `5.3.0`, and
torchdata `0.11.0`. TensorBoard must also be installed for training logs.
The CUDA 12.8 libraries bundled with the Python packages belong to this
virtual environment; installing them does not replace the server's NVIDIA
driver or system CUDA toolkit.

Activate the LF environment before launching (from the LoongForge root):

```bash
source .venv/bin/activate
command -v python torchrun
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Both executable paths should point inside `LoongForge/.venv`, and the CUDA
check should report `2.8.0+cu128 12.8 True`. The original VLDA environment
uses different LeRobot/Transformers versions and is not the LF environment.

Required local artifacts:

- Initial pi05 base `model.safetensors` for stage 1.
- Original V-JEPA2 `vitg.pt`, containing `encoder` and `target_encoder`.
- `paligemma_tokenizer.model` (or a directory containing that file or `tokenizer.model`).
- A LeRobot v2 embedded-image dataset with statistics in `meta/stats.json`.

The dataset reader uses `meta/info.json`, `meta/episodes.jsonl`,
`meta/tasks.jsonl`, and the per-episode parquet files. It supports the existing
LIBERO `image/wrist_image/state/actions` layout and the configured RoboTwin
embedded-image layout. Video storage and LeRobot v3 are not supported by this
strategy.

Statistics follow the LF metadata path: `meta/stats.json` -> `dataset.meta.stats`
-> transforms. State/action entries use the configured `data.state_key` and
`data.action_key`; explicit snapshots may also use `state` and `actions`.
Normalization and action unnormalization use LF's shared `Normalizer`.
Quantile normalization uses `q99 - q01`, substituting epsilon only for a zero
denominator, and does not clamp the normalized values.

The default `q99` mode requires actual `q01/q99` statistics. The inspected
LIBERO `meta/stats.json` only contains mean/std/min/max, so it needs quantile
statistics before using this default. Alternatively, pass
`data.norm_stats_path=/path/to/libero/norm_stats.json` to each new-stage launch
to use the existing quantile file explicitly. This override also accepts a
checkpoint statistics snapshot; it does not depend on any trained VLDA weights.
No implicit fallback to the dataset-root `norm_stats.json` is used.

## LIBERO training

The LIBERO recipe enables pi05's FlexAttention, per-layer compilation, TF32,
and static-graph DDP. The first steps include compilation; compare speed after
warmup. V-JEPA and the VLDA loss definitions are unchanged. For an eager
comparison, append `model.attention_backend=eager model.compile_model=false`.
Set `LOG_INTERVAL=1` to save every step to the console, JSONL and TensorBoard.

In the portable examples below, replace every `/path/to/...` placeholder and
run both `export` lines in the same terminal before launching either stage.

```bash
export DATA_PATH=/path/to/libero
export TOKENIZER_PATH=/path/to/paligemma_tokenizer.model

# Stage 1: world learning from pi05 base; global batch 48, 20k updates.
STAGE=1 GPUS_PER_NODE=4 \
DATA_PATH=/path/to/dataset \
TOKENIZER_PATH=/path/to/tokenizer \
CHECKPOINT_PATH=/path/to/pi05_base_pytorch \
VJEPA_CHECKPOINT=/path/to/vitg.pt \
OUTPUT_DIR=outputs/vlda/libero_stage1 \
bash examples/embodied/vlda/run_vlda_ddp_finetune.sh

# Stage 2: load stage 1 weights; new optimizer/schedule; global batch 16, 160k updates.
STAGE=2 GPUS_PER_NODE=4 \
CHECKPOINT_PATH=outputs/vlda/libero_stage1/checkpoints/steps_20000 \
OUTPUT_DIR=outputs/vlda/libero_stage2 \
bash examples/embodied/vlda/run_vlda_ddp_finetune.sh

# Resume the same stage, with the same batch/worker/schedule configuration.
STAGE=2 RESUME=1 GPUS_PER_NODE=4 \
OUTPUT_DIR=outputs/vlda/libero_stage2 \
bash examples/embodied/vlda/run_vlda_ddp_finetune.sh
```

For the artifacts available in this workspace, this complete stage 1 command
includes all required paths and does not depend on earlier exports:

```bash
STAGE=1 GPUS_PER_NODE=4 \
DATA_PATH=/ssd1/zhanglingsen/datasets/libero \
TOKENIZER_PATH=/ssd1/zhanglingsen/models/pi05/paligemma_tokenizer.model \
CHECKPOINT_PATH=/ssd1/zhanglingsen/models/pi05/pi05_base_pytorch \
VJEPA_CHECKPOINT=/ssd1/zhanglingsen/models/vjepa2/vitg.pt \
OUTPUT_DIR=outputs/vlda/libero_stage1 \
bash examples/embodied/vlda/run_vlda_ddp_finetune.sh \
    data.norm_stats_path=/ssd1/zhanglingsen/datasets/libero/norm_stats.json
```

This dataset's `meta/stats.json` lacks `q01/q99`, so the command explicitly
selects its existing quantile statistics. Supply the same `DATA_PATH`,
`TOKENIZER_PATH`, and statistics override when initializing stage 2; resume
uses the checkpoint's saved statistics.

Set `ZERO_OPTIMIZER=1` to use LoongForge's ZeRO1 optimizer. Global batch equals
`GPUS_PER_NODE * NNODES * per_device_batch_size * GRADIENT_ACCUMULATION_STEPS`.
The launcher derives the per-device batch from `GLOBAL_BATCH_SIZE`.

For a short integration run, override `TRAIN_ITERS`, `WARMUP_ITERS`,
`SAVE_INTERVAL`, and `GLOBAL_BATCH_SIZE` explicitly. For example, use 2 updates
and save every update; these are validation settings, not a finetuning recipe.
As in other LoongForge launchers, trailing CLI and `model.*` / `data.*`
arguments override the defaults.

The LIBERO training budget follows `train_libero_from_base.sh`. State input is
enabled deliberately: the old LIBERO `discrete_state_input=False` setting was
an experiment configuration mistake and is not carried over.

The training recipe matches that script's two stages; this launcher keeps
each stage as a separate command:

| Parameter | Stage 1 | Stage 2 |
| --- | --- | --- |
| Global batch size | 48 | 16 |
| Batch per GPU (4 GPUs, no accumulation) | 12 | 4 |
| Optimizer updates | 20,000 | 160,000 |
| Warmup updates | 1,000 | 10,000 |
| Peak learning rate | 5e-5 | 1e-4 |
| End learning rate | 2e-6 | 1e-5 |
| Cosine schedule length, including warmup | 20,000 | 160,000 |
| Checkpoint interval | 2,000 | 10,000 |
| Metrics interval | 100 | 100 |

Both stages use AdamW with betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay
`1e-10` on all trainable parameters, gradient clipping at `1.0`, seed `42`,
BF16 mixed parameter precision, gradient checkpointing, and teacher EMA
`0.99925`. Warmup starts at `peak_lr / (warmup_steps + 1)`, then cosine decay
runs to the configured end rate. A step is one optimizer update, not an epoch.
Stage 1 freezes the action policy, so its updates do not count as action
finetuning. Stage 2 processes `160000 * 16 = 2.56M` samples.

Runtime settings can be tuned for the LF environment. `NUM_WORKERS` defaults
to `8` per process; adjust it to CPU/storage capacity and measured throughput.
Worker counts do not need to match the original VLDA launcher. When resuming
an existing run, keep its saved worker count because native loader-state
restoration checks this setting.

## Training logs

Each stage saves the following under `OUTPUT_DIR`:

```text
metrics.jsonl                              # Scalar records, appended on resume
tensorboard/events.out.tfevents.*           # TensorBoard curves
logs/train_<timestamp>_node<rank>.log       # Console stdout and stderr
checkpoints/steps_<step>/                   # Weights and training state
```

Metrics are recorded every `LOG_INTERVAL` updates (default `100`). Stage 1
records `loss`, `delta_loss`, and `future_loss`; stage 2 also records
`action_loss` and `align_loss`. LR, gradient norm, teacher EMA, step time,
throughput, and consumed samples are also included. JSONL and TensorBoard
are written by rank 0; console logs are separate for each node and launch.
The pipeline preserves training failures as a nonzero launcher exit code.

Override `LOG_DIR` for console files and `TENSORBOARD_DIR` for event files.
A relative `TENSORBOARD_DIR` is resolved under `OUTPUT_DIR`. For example:

```bash
tensorboard --logdir outputs/vlda
```

Logs use the common LF logger: loss is the current update's mean across GPU
ranks, whereas the original VLDA logger averages rank 0 values over the log
interval. LF uses completed-update numbers (`100`, `200`, ...); the original
logs zero-based update indices (`0`, `100`, ...). LF logs LR after advancing
the scheduler (the next update's LR), whereas the original logs the interval
average of rates already used. The LR used by each optimizer update follows
the same schedule. TensorBoard tag names follow LF (`loss`, `delta_loss`,
`action_loss`, `lr`, etc.), rather than the original `loss/total`, `loss/delta`,
`loss/actions`, and `learning_rate`. These logging conventions do not change
optimization. No W&B account is required.

## Preserved model/data behavior

- Stage 1 freezes PaliGemma and the action expert, and trains online V-JEPA2,
  the world expert, and delta projections. Loss is `delta + future`.
- Stage 2 runs Prefix+World, then Action with differentiable cached prefix
  hidden states. World predictions supply one action token and a noise bias.
  Loss is `delta + future + action + 0.1 * align`.
- Both token-set losses use Sinkhorn OT; only the transport plan is detached.
  Action loss includes all 32 padded action dimensions.
- Future images use `min(t + horizon, episode_end)`. Alignment negatives are
  unique actions in the same episode outside the positive alignment window.
- LIBERO normalizes the 8D robot state, discretizes it into 256 bins, and
  includes it in `Task: ..., State: ...;\nAction: ` before tokenization. State
  padding happens after tokenization, so only the real state dimensions enter
  the prompt. Three camera slots are retained, with the missing right wrist
  masked out.
- V-JEPA receives the base camera only. Each current/future image is repeated
  twice, resized/center-cropped to 256, and ImageNet-normalized. Features remain
  online because the online encoder and teacher change during training.
- Teacher EMA runs once after each optimizer update, including with gradient
  accumulation. It is separate from a policy-wide EMA.
- Eager attention and the original mixed parameter precision are used.
  SigLIP's embedding-to-encoder cast is local to VLDA; no global Transformers
  patch is installed.
- Antialiased image resizing uses PyTorch instead of JAX. On the inspected
  LIBERO samples, uint8 interpolation rounding can differ by one gray level.

## Weights and checkpoints

`--pretrained-checkpoint` loads initial pi05 safetensors for stage 1 and
this implementation's full-model safetensors/pt for stage 2. It maps expert names and old/new
PaliGemma module paths, and rejects missing backbone weights. New world
modules are allowed only when initializing stage 1 from pi05. Stage 2 heads
are allowed to be new only when initializing from stage 1.

Native checkpoints include teacher parameters, optimizer/scheduler/RNG/loader
state, `assets/norm_stats.json`, and `vlda_config.json`. Resume checks model,
data and relevant training settings, world size, metadata fingerprints and
the stats snapshot checksum. Worker random states are included so same-episode
negative sampling can resume through StatefulDataLoader.

The supported workflow is pi05 + pretrained V-JEPA2 -> stage 1 -> stage 2,
with `--resume` restoring checkpoints produced by this implementation. Stage 1
to stage 2 initializes weights and starts a new optimizer. Cross-framework
conversion of the original VLDA optimizer/RNG/loader is outside this workflow.

## RoboTwin

This preset is retained for later work; current validation targets LIBERO.

Use `MODEL_NAME=vlda_robotwin`, provide the original dataset/statistics, and
set training budgets explicitly. This preset uses three cameras, 14 real
action dimensions, a 50-step horizon, state tokens, and Aloha joint/gripper
conversion. Alignment uses the first 10 actions; world prediction still uses
the image at `t+50`, and action loss still covers the full 50-step chunk.

The default PNG cache is `data.vjepa_image_cache_dir=vjepa256/base_0_rgb`.
Override it with an empty string to process the source images online.
The configured RoboTwin dataset was unavailable during migration, so validate
that preset against the actual dataset before a full training run.

## Evaluation

Run the shared eval orchestrator with a Stage 2 checkpoint:

```bash
CHECKPOINT_PATH=/path/to/stage2/checkpoints/steps_80000 \
EVAL_DIR=outputs/vlda/eval_stage2_80000 CUDA_VISIBLE_DEVICES=0 \
bash examples/embodied/vlda/eval/run_libero_eval.sh
```

The launcher prepares the checkpoint config, then manages the server and client
through the same orchestrator as pi05. It defaults to four LIBERO suites with
50 episodes per task. For a smoke run, set `MAX_TASKS=1 EPISODES_PER_TASK=1` and
append `libero_object`. Override `BENCHMARK_PYTHON`, `SERVER_PYTHON`, and
`LIBERO_ROOT` to change the environments. Results are grouped by suite under
`EVAL_DIR`; `PREPARE_ONLY=1` generates the YAML without starting evaluation.

Alternatively, fill the paths in `eval/configs/libero/libero.yaml` and run:

```bash
CONFIG=examples/embodied/vlda/eval/configs/libero/libero.yaml \
CUDA_VISIBLE_DEVICES=0 bash examples/embodied/vlda/eval/run_libero_eval.sh
```

With `CONFIG`, preparation is skipped and evaluation settings/output paths come
from YAML. Both routes default to client seed `7` with server seeding disabled.

The factory loads the checkpoint's data settings and stats, then removes training-only modules.
Inference uses three world steps followed by ten action steps and requires no
future observation or V-JEPA encoder. The LIBERO payload supplies the same 8D
state layout as training: end-effector position, axis-angle rotation, and both
gripper finger positions. Rollout validation is deferred.

For RoboTwin, set `benchmark.action_bridge: vlda_joint14` and the matching
model dimensions/horizon. This bridge sends the raw joint state and executes
the returned absolute joints. State conversion and the inverse action
conversion already happen in the policy; do not use `aloha_pi` input encoding
or the pi05 Aloha action decoder a second time.

## Validation and current limits

```bash
python -m pytest tests/embodied/unit/test_vlda.py -q
```

Tests use actual PiGemma/SigLIP/V-JEPA layers at small dimensions: both stages,
gradient checkpointing, mixed precision, teacher EMA, alignment gradients,
OT permutation behavior, pi05 initialization, native stage transition,
strict resume, inference cache isolation, episode boundaries, 0/2-worker loader
restore, and two-process DDP/ZeRO1 with gradient accumulation. Data checks cover
metadata/explicit statistics, state-conditioned tokens, LF denominator behavior,
action unnormalization, and the statistics snapshot used during resume.
They do not require pretrained weights or a GPU.

A separate comparison with the original VLDA implementation used identical
small-model weights and inputs. All four losses and the checked gradients
(world/action projections, world-to-action token, and online encoder patch
embedding) matched exactly in both FP32 and authored BF16 precision.
That comparison covers core model math; the LIBERO state configuration and
normalization now intentionally follow the choices described above.

The local Python 3.12 environment with PyTorch 2.8.0+cu128, LeRobot 0.5.0,
and Transformers 5.3.0 passes the VLDA tests without a dependency overlay.
Training-entry imports, a real LIBERO sample with state tokens, JSONL and
TensorBoard output, and a CUDA tensor computation have also been checked.
VLDA applies its transform directly so Python 3.12 can recognize the dataset's
state-saving methods and restore worker randomness through StatefulDataLoader.

A full-size four-GPU stage 1 run has reached the training loop and emitted
step timing records. Completed training and benchmark rollout success remain
to be validated. This implementation supports DDP and DDP+ZeRO1;
FSDP, LoRA, and compiled attention are not enabled. FSDP needs explicit teacher
shard alignment, trainable-scalar handling and wrapping of the actual custom
forward call boundaries.
