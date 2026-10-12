# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Prepare checkpoint-specific VLDA settings for the shared LIBERO evaluator."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from loongforge.embodied.model.vlda.model_configuration_vlda import VLDAModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=12093)
    parser.add_argument("--health-port", type=int, default=12094)
    parser.add_argument("--seed", type=int, default=7, help="LIBERO client seed")
    parser.add_argument("--server-seed", type=int, default=None, help="Optional model server seed; unset by default")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "suites", nargs="*", default=["libero_spatial", "libero_object", "libero_10", "libero_goal"],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    checkpoint = args.checkpoint_path.resolve()
    output = args.output_dir.resolve()
    tokenizer = args.tokenizer_path.resolve()
    stats = checkpoint / "assets/norm_stats.json"
    for path in (checkpoint / "vlda_config.json", stats, tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not any((checkpoint / name).is_file() for name in ("model.safetensors", "pytorch_model.pt")):
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.pt in {checkpoint}")
    manifest = json.loads((checkpoint / "vlda_config.json").read_text())
    if manifest["model"]["stage"] != 2:
        raise ValueError("LIBERO evaluation requires a Stage 2 checkpoint")

    config = yaml.safe_load(Path(__file__).with_name("configs").joinpath("libero/libero.yaml").read_text())
    # Omit unchanged defaults, including OmegaConf's read-only nested V-JEPA config.
    defaults = asdict(VLDAModelConfig())
    # Automatic mode takes model fields from the checkpoint, not manual YAML overrides.
    config["model"] = {key: value for key, value in config["model"].items() if key not in defaults}
    config["model"].update({key: value for key, value in manifest["model"].items() if value != defaults[key]})
    config["model"].update(
        model_type="vlda", name="loongforge-vlda-libero", gradient_checkpointing=False,
        compile_model=False, attention_backend="eager", vjepa_checkpoint="",
    )
    config["benchmark"].update(
        max_tasks=args.max_tasks, episodes_per_task=args.episodes_per_task, max_steps=args.max_steps,
        prefer_controller_ee=False, control_mode="delta",
    )
    config["server"].update(
        ckpt_path=str(checkpoint), tokenizer_path=str(tokenizer), dataset_statistics_path=str(stats),
        python=sys.executable, loongforge_root=str(root), device="cuda",
        port=args.port, health_port=args.health_port, log=str(output / "server.log"),
        seed=args.server_seed, float32_matmul_precision="high",
    )
    config["env"].update(
        eval_root=str(root / "loongforge/embodied/eval"), loongforge_root=str(root),
        libero_config_path=str(output / "libero_config"),
        mujoco_gl=os.environ.get("MUJOCO_GL", "egl"), pyopengl_platform=os.environ.get("PYOPENGL_PLATFORM", "egl"),
    )
    config["run"].update(output_dir=str(output), seed=args.seed, timestamped_output=False)
    config["sweep"] = [
        {"benchmark.suite": suite, "run.id": suite, "server.log": str(output / suite / "server.log")}
        for suite in args.suites
    ]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Checkpoint: {checkpoint}\nConfig: {config_path}\nSuites: {', '.join(args.suites)}")


if __name__ == "__main__":
    main()
