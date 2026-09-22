# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA migration checks using real PiGemma/V-JEPA layers at small dimensions."""

import io
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("lerobot.policies.pi_gemma")

from loongforge.embodied.data.datasets.transforms.utils.normalizer import Normalizer
from loongforge.embodied.data.datasets.vlda import VLDASampler, build_vlda_sampler
from loongforge.embodied.data.datasets.vlda.transforms.data_configuration_vlda import VLDADataConfig
from loongforge.embodied.data.datasets.vlda.transforms.vlda_collator import VLDAPreparedBatch
from loongforge.embodied.data.datasets.vlda.transforms.vlda_transform import (
    VLDATransform,
    decode_actions,
    encode_actions,
    encode_state,
)
from loongforge.embodied.data.datasets.vlda.vlda_dataset import VLDADataset
from loongforge.embodied.model.vlda import modeling_vlda, vlda_gemma
from loongforge.embodied.model.vlda.model_configuration_vlda import VJEPA2Config, VLDAModelConfig


def _tiny_policy_builder(monkeypatch):
    torch.set_num_threads(1)
    dimensions = SimpleNamespace(width=16, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=16, depth=2)
    monkeypatch.setattr(modeling_vlda, "_get_gemma_config", lambda variant: dimensions)
    paligemma_cls = vlda_gemma.PaliGemmaForConditionalGeneration
    gemma_cls = vlda_gemma.GemmaForCausalLM

    def small_paligemma(config):
        config.text_config.vocab_size = 32
        config.image_token_index = 31
        config.vision_config.hidden_size = 16
        config.vision_config.projection_dim = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        config.vision_config.image_size = 16
        config.vision_config.patch_size = 8
        return paligemma_cls(config)

    def small_gemma(config):
        config.vocab_size = 32
        return gemma_cls(config)

    monkeypatch.setattr(vlda_gemma, "PaliGemmaForConditionalGeneration", small_paligemma)
    monkeypatch.setattr(vlda_gemma, "GemmaForCausalLM", small_gemma)

    def preprocessed(self, observation, *, train=True):
        # Isolate model math from the separately tested production 224/256 resize.
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            list(observation.vjepa_image.values()) if observation.vjepa_image else None,
            list(observation.vjepa_future_image.values()) if observation.vjepa_future_image else None,
        )

    monkeypatch.setattr(modeling_vlda.VLDAPytorch, "_preprocess_observation", preprocessed)

    def build(stage, checkpointing=False, dtype="float32", **kwargs):
        config = VLDAModelConfig(
            stage=stage,
            action_dim=3,
            state_dim=3,
            max_action_dim=4,
            max_state_dim=4,
            action_horizon=2,
            delta_token_num=4,
            delta_latent_dim=8,
            contrast_dim=8,
            dtype=dtype,
            gradient_checkpointing=checkpointing,
            vjepa=VJEPA2Config(img_size=16, patch_size=8, embed_dim=24, depth=1, num_heads=2),
            **kwargs,
        )
        torch.manual_seed(7)
        return modeling_vlda.VLDAPolicy(config)

    return build


@pytest.fixture
def tiny_policy(monkeypatch):
    return _tiny_policy_builder(monkeypatch)


def model_batch():
    torch.manual_seed(13)
    return VLDAPreparedBatch(
        images={"base_0_rgb": torch.randn(2, 3, 16, 16)},
        image_masks={"base_0_rgb": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros(2, 4),
        tokenized_prompt=torch.tensor([[1, 2, 3], [1, 2, 0]]),
        tokenized_prompt_mask=torch.tensor([[True, True, True], [True, True, False]]),
        actions=torch.randn(2, 2, 4),
        neg_actions=torch.randn(2, 8, 4),
        vjepa_image={"base_0_rgb": torch.randn(2, 3, 2, 16, 16)},
        vjepa_future_image={"base_0_rgb": torch.randn(2, 3, 2, 16, 16)},
    )


@pytest.mark.parametrize("stage", [1, 2])
def test_training_module_initialization_is_seeded_and_preserves_base(tiny_policy, stage):
    core = tiny_policy(stage).model
    initial = {name: value.clone() for name, value in core.state_dict().items()}
    rng_state = torch.get_rng_state()
    core.initialize_training_modules(42)
    assert torch.equal(torch.get_rng_state(), rng_state)
    initialized = {name: value.clone() for name, value in core.state_dict().items()}
    protected = (
        "paligemma_with_expert.paligemma.",
        "paligemma_with_expert.action_gemma_expert.",
        "vjepa_encoder.",
        "teacher_vjepa_encoder.",
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    )
    for name, value in initialized.items():
        if name.startswith(protected):
            assert torch.equal(value, initial[name]), name
    world = core.paligemma_with_expert.world_gemma_expert.model
    for norm in [world.norm, *(layer.input_layernorm for layer in world.layers)]:
        assert torch.count_nonzero(norm.dense.weight) > 0
        assert torch.count_nonzero(norm.dense.bias) == 0
    if stage == 2:
        assert torch.count_nonzero(core.delta_to_action_noise[3].weight) == 0
        assert torch.count_nonzero(core.delta_to_action_noise[3].bias) == 0

    torch.manual_seed(123)
    torch.rand(17)
    core.initialize_training_modules(42)
    for name, value in core.state_dict().items():
        assert torch.equal(value, initialized[name]), name
    core.initialize_training_modules(43)
    assert not torch.equal(core.world_in_proj.weight, initialized["world_in_proj.weight"])
    assert not torch.equal(
        world.layers[0].self_attn.q_proj.weight,
        initialized["paligemma_with_expert.world_gemma_expert.model.layers.0.self_attn.q_proj.weight"],
    )


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_stage_loss_gradients_and_teacher(tiny_policy, stage, checkpointing):
    policy = tiny_policy(stage, checkpointing)
    loss, metrics = policy(model_batch())
    assert torch.isfinite(loss)
    expected = metrics["delta_loss"] + metrics["future_loss"]
    if stage == 2:
        expected += metrics["action_loss"] + 0.1 * metrics["align_loss"]
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    core = policy.model
    assert all(parameter.grad is None for parameter in core.teacher_vjepa_encoder.parameters())
    assert any(parameter.grad is not None for parameter in core.vjepa_encoder.parameters())
    assert any(parameter.grad is not None for parameter in core.paligemma_with_expert.world_gemma_expert.parameters())
    prefix = core.paligemma_with_expert.paligemma
    if stage == 1:
        assert all(not parameter.requires_grad for parameter in prefix.parameters())
        assert core.world_to_action_token is None
    else:
        assert prefix.model.language_model.embed_tokens.weight.grad.abs().sum() > 0
        assert core.world_to_action_token[1].weight.grad.abs().sum() > 0
    before = [parameter.detach().clone() for parameter in core.teacher_vjepa_encoder.parameters()]
    optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=1e-3)
    optimizer.step()
    policy.update_teacher_ema(0.9)
    for old, online, teacher in zip(before, core.vjepa_encoder.parameters(), core.teacher_vjepa_encoder.parameters()):
        torch.testing.assert_close(teacher, old * 0.9 + online * 0.1)


def test_alignment_does_not_update_action_projection(tiny_policy):
    core = tiny_policy(2).model
    world = torch.randn(2, 4, 8, requires_grad=True)
    batch = model_batch()
    core.compute_delta_action_align_loss(world, batch.actions, batch.neg_actions).backward()
    assert core.action_in_proj.weight.grad is None
    assert world.grad.abs().sum() > 0
    assert core.action_to_align[1].weight.grad.abs().sum() > 0


def test_authored_mixed_precision_forward_backward(tiny_policy):
    policy = tiny_policy(2, dtype="bfloat16")
    loss, _ = policy(model_batch())
    loss.backward()
    vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    assert vision.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vision.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert policy.model.action_in_proj.weight.dtype == torch.float32
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_compiled_flex_preserves_losses_and_gradients(tiny_policy, stage, dtype):
    torch._dynamo.reset()
    reference = tiny_policy(stage, checkpointing=True, dtype=dtype).cuda()
    optimized = tiny_policy(
        stage, checkpointing=True, dtype=dtype, attention_backend="flex", compile_model=True,
    ).cuda()
    optimized.load_state_dict(reference.state_dict(), strict=True)
    batch = model_batch().to("cuda")
    results = []
    for policy in (reference, optimized):
        torch.manual_seed(42)
        loss, metrics = policy(batch)
        loss.backward()
        gradients = {
            name: param.grad.detach().float().flatten()
            for name, param in policy.named_parameters() if param.grad is not None
        }
        results.append((metrics, gradients))
    expected, actual = results
    for key in expected[0]:
        torch.testing.assert_close(actual[0][key], expected[0][key], rtol=0.01, atol=1e-5)
    # AOTAutograd may materialize a zero for an unused output of the final LLM layer.
    for key in actual[1].keys() ^ expected[1].keys():
        gradient = actual[1].get(key, expected[1].get(key))
        assert torch.count_nonzero(gradient) == 0, key
        actual[1].pop(key, None)
        expected[1].pop(key, None)
    expected_grad = torch.cat(list(expected[1].values()))
    actual_grad = torch.cat(list(actual[1].values()))
    relative_error = (actual_grad - expected_grad).norm() / expected_grad.norm()
    assert relative_error < 0.01, relative_error.item()


def test_trainer_accumulation_updates_teacher_once(tiny_policy):
    from loongforge.embodied.train.trainers.custom.vlda import VLDAFinetuneTrainer
    from loongforge.embodied.train.training_args import TrainingArgs
    from loongforge.embodied.train.utils.logging import StageTimers

    policy = tiny_policy(2)
    args = TrainingArgs(gradient_accumulation_steps=2, check_for_nan_in_loss_and_grad=False)
    trainer = VLDAFinetuneTrainer(args, policy.config, VLDADataConfig())
    trainer.model = policy
    trainer.ctx = SimpleNamespace(is_distributed=False)
    trainer._stage_timers = StageTimers()
    trainer._fetch_batch = lambda name: model_batch()
    trainer.optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad])
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.0)
    original = [parameter.detach().clone() for parameter in policy.model.teacher_vjepa_encoder.parameters()]
    trainer._train_step()
    for before, teacher in zip(original, policy.model.teacher_vjepa_encoder.parameters()):
        torch.testing.assert_close(before, teacher, rtol=0, atol=0)
    trainer.completed_steps = 1
    metrics = {}
    trainer._on_step_end(metrics)
    momentum = policy.config.ema_start
    for before, online, teacher in zip(
        original, policy.model.vjepa_encoder.parameters(), policy.model.teacher_vjepa_encoder.parameters()
    ):
        torch.testing.assert_close(teacher, momentum * before + (1 - momentum) * online)
    assert metrics["teacher_ema"] == momentum


def test_ot_permutation_and_target_gradients(tiny_policy):
    core = tiny_policy(1).model
    prediction = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, requires_grad=True)
    loss = core.compute_token_set_mse_loss(prediction, target)
    permuted = core.compute_token_set_mse_loss(prediction[:, [2, 0, 3, 1]], target[:, [3, 1, 0, 2]])
    torch.testing.assert_close(loss, permuted)
    loss.backward()
    assert prediction.grad.abs().sum() > 0
    assert target.grad.abs().sum() > 0


def test_inference_preserves_prefix_cache_and_is_repeatable(tiny_policy, monkeypatch):
    policy = tiny_policy(2).eval()
    core = policy.model
    original = core.paligemma_with_expert.forward
    cache_lengths = []

    def check_cache(**kwargs):
        cache = kwargs.get("past_key_values")
        if cache is not None:
            cache_lengths.append(cache.get_seq_length())
        return original(**kwargs)

    monkeypatch.setattr(core.paligemma_with_expert, "forward", check_cache)
    batch = model_batch()
    batch.vjepa_image = batch.vjepa_future_image = None
    core.drop_training_only_modules()
    noise = torch.randn(2, 2, 4)
    world_noise = torch.randn(2, 4, 8)
    first = policy.predict_action_chunk(batch, noise=noise, world_noise=world_noise)
    second = policy.predict_action_chunk(batch, noise=noise, world_noise=world_noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 2, 3)
    assert len(cache_lengths) == 2 * (3 + 10)
    assert len(set(cache_lengths)) == 1


def test_native_stage_transition_and_strict_resume(tiny_policy, tmp_path):
    from safetensors.torch import save_model

    stage1 = tiny_policy(1)
    path = tmp_path / "stage1.safetensors"
    save_model(stage1, str(path))
    stage2 = tiny_policy(2)
    stage2.load_pretrained(path)
    torch.testing.assert_close(stage2.model.world_in_proj.weight, stage1.model.world_in_proj.weight)
    assert torch.count_nonzero(stage2.model.delta_to_action_noise[3].weight) == 0
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        stage2.load_pretrained(path, resume=True)
    save_model(stage2, str(tmp_path / "full.safetensors"))
    restored = tiny_policy(2)
    restored.load_pretrained(tmp_path / "full.safetensors", resume=True)
    for key, value in stage2.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_pi05_initialization_loads_both_vjepa_encoders(tiny_policy, tmp_path):
    from safetensors.torch import save_file

    source = tiny_policy(1)
    base = {
        key.replace(".action_gemma_expert.", ".gemma_expert.")
        .replace("time_mlp_in.", "action_time_mlp_in.")
        .replace("time_mlp_out.", "action_time_mlp_out."): value.clone()
        for key, value in source.model.state_dict().items()
        if not key.startswith(modeling_vlda._WORLD_PREFIXES)
    }
    path = tmp_path / "pi05.safetensors"
    save_file(base, str(path))
    encoder = source.model.vjepa_encoder.encoder.state_dict()
    teacher = {f"module.backbone.{key}": value + 0.25 for key, value in encoder.items()}
    encoder_path = tmp_path / "vjepa.pt"
    torch.save({"encoder": encoder, "target_encoder": teacher}, encoder_path)
    destination = tiny_policy(1)
    destination.config = replace(destination.config, vjepa_checkpoint=str(encoder_path))
    destination.load_pretrained(path)
    for key, value in encoder.items():
        torch.testing.assert_close(destination.model.vjepa_encoder.encoder.state_dict()[key], value)
        torch.testing.assert_close(destination.model.teacher_vjepa_encoder.encoder.state_dict()[key], value + 0.25)
    base.pop("action_in_proj.weight")
    save_file(base, str(path))
    with pytest.raises(ValueError, match="action_in_proj.weight"):
        destination.load_pretrained(path)


def _ddp_worker(rank, rendezvous, zero, accelerated=False):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from loongforge.embodied.optimizer import build_optimizer
    from loongforge.embodied.train.trainers.custom.vlda import VLDAFinetuneTrainer
    from loongforge.embodied.train.training_args import TrainingArgs
    from loongforge.embodied.train.utils.logging import StageTimers

    with pytest.MonkeyPatch.context() as patches:
        tiny_policy = _tiny_policy_builder(patches)
        device = torch.device("cuda", rank) if accelerated else torch.device("cpu")
        if accelerated:
            torch.cuda.set_device(device)
            torch._dynamo.config.optimize_ddp = False
        dist.init_process_group(
            "nccl" if accelerated else "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        )
        try:
            policy = tiny_policy(
                2, checkpointing=True, dtype="bfloat16", attention_backend="flex" if accelerated else "eager",
                compile_model=accelerated,
            ).to(device)
            args = TrainingArgs(
                optimizer="AdamW",
                distributed_strategy="ddp",
                zero_optimizer=zero,
                gradient_accumulation_steps=2,
                check_for_nan_in_loss_and_grad=False,
            )
            trainer = VLDAFinetuneTrainer(args, policy.config, VLDADataConfig())
            trainer.model = DistributedDataParallel(
                policy, find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=True,
                device_ids=[rank] if accelerated else None,
            )
            trainer.ctx = SimpleNamespace(is_distributed=True)
            trainer._stage_timers = StageTimers()

            def fetch(name):
                batch = model_batch()
                batch.actions += rank
                return batch.to(device)

            trainer._fetch_batch = fetch
            trainer.optimizer = build_optimizer(trainer.model, args)
            trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.0)
            for step in range(2):
                metrics, norm = trainer._train_step()
                assert torch.isfinite(torch.as_tensor(norm))
                trainer.completed_steps = step + 1
                trainer._on_step_end(metrics)
            for parameter in policy.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("zero", [False, True])
def test_two_rank_ddp_with_mixed_precision_and_accumulation(tmp_path, zero):
    import torch.multiprocessing as mp

    mp.spawn(_ddp_worker, args=(str(tmp_path / "rendezvous"), zero), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Compiled DDP requires two GPUs")
def test_compiled_flex_ddp_with_accumulation(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_ddp_worker, args=(str(tmp_path / "rendezvous"), False, True), nprocs=2, join=True)


@pytest.fixture
def episode_dataset(tmp_path):
    import pandas as pd

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    info = {
        "codebase_version": "v2.0",
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "features": {"image": {"dtype": "image"}, "wrist_image": {"dtype": "image"}},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": i, "length": 12}) for i in range(2))
    )
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick_object"}) + "\n")
    stats = {key: {"q01": [-1.0] * 3, "q99": [1.0] * 3} for key in ("state", "actions")}
    (root / "meta/stats.json").write_text(json.dumps(stats))
    for episode in range(2):
        rows = []
        for frame in range(12):
            buffer = io.BytesIO()
            Image.fromarray(np.full((16, 16, 3), frame + 20 * episode, dtype=np.uint8)).save(buffer, format="PNG")
            image = {"bytes": buffer.getvalue(), "path": None}
            rows.append(
                {
                    "image": image,
                    "wrist_image": image,
                    "state": np.full(3, frame, dtype=np.float32),
                    "actions": np.full(3, frame + 20 * episode, dtype=np.float32),
                    "task_index": 0,
                    "episode_index": episode,
                }
            )
        pd.DataFrame(rows).to_parquet(root / f"data/episode_{episode:06d}.parquet")
    cfg = VLDAModelConfig(stage=2, action_dim=3, state_dim=3, action_horizon=2)
    return VLDADataset(root, cfg, VLDADataConfig())


@pytest.fixture
def tokenizer_path(tmp_path):
    import sentencepiece

    prefix = str(tmp_path / "tokenizer")
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["Task: pick object, State: 128 0 255; Action:"]),
        model_prefix=prefix,
        model_type="char",
        vocab_size=64,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return prefix + ".model"


def test_metadata_statistics_and_explicit_override(episode_dataset):
    root = episode_dataset.root
    alternate = {key: {"q01": [-2.0] * 3, "q99": [2.0] * 3} for key in ("state", "actions")}
    path = root / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": alternate}))
    default = VLDADataset(root, episode_dataset.model_cfg, VLDADataConfig())
    override = VLDADataset(root, episode_dataset.model_cfg, VLDADataConfig(norm_stats_path=str(path)))
    np.testing.assert_array_equal(default.meta.stats["actions"]["q99"], [1.0] * 3)
    np.testing.assert_array_equal(override.meta.stats["actions"]["q99"], [2.0] * 3)


def test_state_prompt_and_lf_normalization(episode_dataset, tokenizer_path):
    # The tiny and zero quantile ranges distinguish LF's denominator behavior
    # from the previous unconditional +1e-6 formula.
    stats = {key: {"q01": np.zeros(3), "q99": np.array([1.0, 0.0, 1e-7])} for key in ("observation.state", "action")}
    data_cfg = VLDADataConfig(state_key="observation.state", action_key="action")
    transform = VLDATransform(episode_dataset.model_cfg, data_cfg, stats, tokenizer_path)
    sample = episode_dataset[0]
    sample["state"] = np.array([0.5, 0.0, 1e-7])
    sample["actions"][:] = sample["state"]
    sample["neg_actions"][:] = sample["state"]
    prepared = transform(sample)
    expected = torch.tensor([0.0, -1.0, 1.0])
    torch.testing.assert_close(prepared["state"][:3], expected, rtol=0, atol=0)
    torch.testing.assert_close(prepared["actions"][0, :3], expected, rtol=0, atol=0)
    torch.testing.assert_close(prepared["neg_actions"][0, :3], expected, rtol=0, atol=0)
    assert torch.count_nonzero(prepared["state"][3:]) == 0
    assert torch.count_nonzero(prepared["actions"][:, 3:]) == 0
    expected_tokens = transform.tokenizer.processor.encode(
        "Task: pick object, State: 128 0 255;\nAction: ", add_bos=True
    )
    assert prepared["tokenized_prompt"][prepared["tokenized_prompt_mask"]].tolist() == expected_tokens
    sample["state"][0] = 0
    changed = transform(sample)
    assert not torch.equal(changed["tokenized_prompt"], prepared["tokenized_prompt"])


def test_q99_requires_actual_quantiles(episode_dataset, tokenizer_path):
    stats = {key: {"min": np.zeros(3), "max": np.ones(3)} for key in ("state", "actions")}
    with pytest.raises(ValueError, match="data.norm_stats_path"):
        VLDATransform(episode_dataset.model_cfg, VLDADataConfig(), stats, tokenizer_path)


def test_inference_uses_state_and_lf_inverse(tiny_policy, tokenizer_path, monkeypatch):
    policy = tiny_policy(2)
    stats = {key: {"q01": np.zeros(3), "q99": np.array([2.0, 0.0, 1e-7])} for key in ("state", "actions")}
    policy._inference_transform = VLDATransform(policy.config, VLDADataConfig(image_size=16), stats, tokenizer_path)
    captured = {}

    def predict(batch):
        captured["batch"] = batch
        return torch.tensor([[[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])

    monkeypatch.setattr(policy, "predict_action_chunk", predict)
    images = [[np.zeros((16, 16, 3), dtype=np.uint8)]]
    with pytest.raises(ValueError, match="requires the current robot state"):
        policy.predict_action(images, ["pick object"])
    actions = policy.predict_action(images, ["pick object"], state=[1.0, 0.0, 1e-7])
    np.testing.assert_allclose(actions, [[[0.0, 0.0, 1e-7], [1.0, 0.0, 5e-8]]], rtol=1e-6, atol=0)
    torch.testing.assert_close(captured["batch"].state[0, :3], torch.tensor([0.0, -1.0, 1.0]))


def test_libero_payload_supplies_eight_dimensional_state():
    from loongforge.embodied.eval.payload_builders.vlda import VLDAPayloadBuilder

    canonical = {
        "images": {"primary": np.zeros((16, 16, 3), dtype=np.uint8)},
        "instruction": "pick object",
        "state_raw": {
            "eef_pos": [0.1, 0.2, 0.3],
            "eef_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_qpos": [0.04, -0.04],
        },
    }
    payload = VLDAPayloadBuilder().build(canonical, {})
    np.testing.assert_allclose(payload["state"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04])


def test_future_tail_padding_and_same_episode_negatives(episode_dataset):
    for index in (0, 10, 11, 12, 23):
        sample = episode_dataset[index]
        start = index // 12 * 12
        assert sample["future_index"] == min(index + 2, start + 11)
        assert np.all((sample["negative_indices"] >= start) & (sample["negative_indices"] < start + 12))
        assert not any(index <= negative < index + 2 for negative in sample["negative_indices"])
        assert len(set(sample["negative_indices"])) == 8
    np.testing.assert_array_equal(episode_dataset[11]["actions"][0], episode_dataset[11]["actions"][1])


def _sample_indices(samples):
    return [(sample["index"], sample["negative_indices"].tolist()) for sample in samples]


def _identity_sample(sample):
    return sample


@pytest.mark.parametrize("world_size", [1, 4])
def test_vlda_sampler_matches_original_order(world_size):
    from torch.utils.data.distributed import DistributedSampler

    from loongforge.embodied.data.datasets.sampler_builder import SamplerBuilderContext

    dataset = range(101)
    for rank in range(world_size):
        sampler = build_vlda_sampler(
            SamplerBuilderContext(
                dataset=dataset,
                training_args=SimpleNamespace(distributed_sampler_mode="cyclic", batch_drop_last=True),
                ctx=SimpleNamespace(world_size=world_size, rank=rank),
                batch_size=2,
                seed=42,
                shuffle=True,
            )
        )
        original = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=42, drop_last=True)
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            original.set_epoch(epoch)
            assert list(sampler) == [(index, epoch) for index in original]


def test_negatives_depend_on_sample_seed_and_epoch(episode_dataset):
    sample = episode_dataset[(5, 0)]
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    np.testing.assert_array_equal(sample["negative_indices"], episode_dataset[5]["negative_indices"])
    assert not np.array_equal(sample["negative_indices"], episode_dataset[(5, 1)]["negative_indices"])
    other_seed = VLDADataset(episode_dataset.root, episode_dataset.model_cfg, episode_dataset.data_cfg, seed=43)
    assert not np.array_equal(sample["negative_indices"], other_seed[5]["negative_indices"])


def test_negative_sampling_worker_independence_and_epoch_resume(episode_dataset):
    from torchdata.stateful_dataloader import StatefulDataLoader

    def build(workers):
        return StatefulDataLoader(
            episode_dataset,
            batch_size=2,
            sampler=VLDASampler(episode_dataset, num_replicas=1, rank=0, seed=42, drop_last=True),
            num_workers=workers,
            collate_fn=_sample_indices,
            generator=torch.Generator().manual_seed(42),
            multiprocessing_context="spawn" if workers else None,
            persistent_workers=workers > 0,
        )

    reference = build(0)
    loader = build(2)
    for epoch in (0, 1):
        reference.sampler.set_epoch(epoch)
        loader.sampler.set_epoch(epoch)
        expected = list(reference)
        iterator = iter(loader)
        assert [next(iterator), next(iterator)] == expected[:2]
        snapshot = loader.state_dict()
        assert list(iterator) == expected[2:]
        restored = build(2)
        restored.load_state_dict(snapshot)
        assert list(restored) == expected[2:]
        del restored


def test_training_seed_reset_preserves_resume_rng():
    from loongforge.embodied.train.trainers.custom.vlda.vlda_finetune_trainer import VLDAFinetuneTrainer
    from loongforge.embodied.train.utils.utils import set_seed

    trainer = object.__new__(VLDAFinetuneTrainer)
    trainer.training_args = SimpleNamespace(seed=42, set_seed_by_rank=False, resume=False)
    set_seed(42)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(123)
    trainer._on_after_data_iterators_initialized()
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    trainer.training_args.resume = True
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    trainer._on_after_data_iterators_initialized()
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert after_numpy[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_restores_negative_sampling(episode_dataset, workers):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from loongforge.embodied.data.dataloader import _apply_transform

    def build():
        return StatefulDataLoader(
            _apply_transform(episode_dataset, _identity_sample),
            batch_size=2,
            shuffle=False,
            num_workers=workers,
            collate_fn=_sample_indices,
            generator=torch.Generator().manual_seed(42),
            multiprocessing_context="spawn" if workers else None,
        )

    random.seed(42)
    loader = build()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    snapshot = loader.state_dict()
    expected = [next(iterator), next(iterator)]
    restored = build()
    restored.load_state_dict(snapshot)
    resumed = iter(restored)
    assert [next(resumed), next(resumed)] == expected


def test_aloha_action_conversion_and_normalization_round_trip():
    config = VLDADataConfig(adapt_to_pi=True, delta_joint_actions=True)
    state = encode_state(np.linspace(0.1, 0.8, 14), True)
    actions = np.linspace(-0.3, 0.9, 42).reshape(3, 14)
    stats = {"q01": np.full(14, -1.0), "q99": np.full(14, 2.0)}
    encoded = encode_actions(actions, state, config)
    normalizer = Normalizer("q99", stats)
    restored = normalizer.inverse(normalizer(torch.as_tensor(encoded, dtype=torch.float32))).numpy()
    np.testing.assert_allclose(decode_actions(restored, state, config), actions, atol=1e-6)


def test_robotwin_bridge_preserves_raw_joints_and_action_order(monkeypatch):
    from loongforge.embodied.eval import transport
    from loongforge.embodied.eval.bridges.robotwin_policy import ModelClient

    captured = {}
    action = np.arange(14, dtype=np.float32) / 20

    class Client:
        def reset(self, episode):
            pass

        def predict_action(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True, "data": {"actions": action.tolist()}}

    monkeypatch.setattr(transport, "PolicyClient", lambda **kwargs: Client())
    client = ModelClient(action_bridge="vlda_joint14")
    joints = np.linspace(0.1, 0.9, 14, dtype=np.float32)
    observation = {
        "observation": {
            name: {"rgb": np.full((16, 16, 3), value, dtype=np.uint8)}
            for name, value in (("head_camera", 10), ("left_camera", 20), ("right_camera", 30))
        },
        "joint_action": {"vector": joints},
    }
    result = client.step(observation, instruction="pick the object")
    np.testing.assert_array_equal(captured["state"], joints)
    np.testing.assert_array_equal(result, action)


def test_native_checkpoint_assets_and_resume(tiny_policy, episode_dataset, tmp_path):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from loongforge.embodied.distributed.checkpoint import resume_training_state
    from loongforge.embodied.train.trainers.custom.vlda import VLDAFinetuneTrainer
    from loongforge.embodied.train.training_args import TrainingArgs

    policy = tiny_policy(2)
    args = TrainingArgs(save_format="safetensors", distributed_strategy="ddp", output_dir=str(tmp_path))
    trainer = VLDAFinetuneTrainer(args, policy.config, VLDADataConfig())
    trainer.model = policy
    trainer.ctx = SimpleNamespace(
        is_main=True, is_distributed=False, world_size=1, rank=0, device=torch.device("cpu"), barrier=lambda: None
    )
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    trainer.optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad])
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.0)
    trainer.dataloaders = {"vla": StatefulDataLoader(episode_dataset, batch_size=2, collate_fn=_sample_indices)}
    trainer._data_iters["vla"] = iter(trainer.dataloaders["vla"])
    next(trainer._data_iters["vla"])
    loss, _ = policy(model_batch())
    loss.backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.completed_steps = 1
    trainer._on_step_end({})
    trainer._save_checkpoint()
    path = tmp_path / "checkpoints/steps_1"
    assert (path / "training_state.pt").is_file()
    assert (path / "assets/norm_stats.json").is_file()
    fresh = VLDAFinetuneTrainer(replace(args, resume=True), policy.config, VLDADataConfig())
    fresh.model = tiny_policy(2)
    fresh.ctx = trainer.ctx
    fresh.logger = SimpleNamespace(log_resume=lambda step: None, log_pretrained_loaded=lambda path: None)
    fresh._handle_resume(path, 1, 0)
    assert fresh.completed_steps == 1
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, fresh.model.state_dict()[key])
    fresh.optimizer = torch.optim.AdamW(
        [parameter for parameter in fresh.model.parameters() if parameter.requires_grad]
    )
    fresh.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh.optimizer, lambda step: 1.0)
    epoch, loader_state, rng_state = resume_training_state(
        fresh.model, fresh.optimizer, fresh.lr_scheduler, path, fresh.ctx, restore_rng=False
    )
    assert epoch == trainer.current_epoch
    assert loader_state["vla"]
    assert len(rng_state) == 1
    assert fresh.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()
    saved = trainer.optimizer.state_dict()["state"]
    restored = fresh.optimizer.state_dict()["state"]
    assert saved.keys() == restored.keys()
    for parameter_id, values in saved.items():
        for name, value in values.items():
            torch.testing.assert_close(value, restored[parameter_id][name])
    # Metadata may change or disappear after saving; resume owns its stats.
    (episode_dataset.root / "meta/stats.json").unlink()
    resumed_dataset = VLDADataset(episode_dataset.root, fresh.model_cfg, fresh.data_cfg)
    for key, values in episode_dataset.meta.stats.items():
        for name, value in values.items():
            np.testing.assert_array_equal(resumed_dataset.meta.stats[key][name], value)
    snapshot = path / "assets/norm_stats.json"
    snapshot.write_text(snapshot.read_text() + " ")
    with pytest.raises(ValueError, match="normalization statistics changed"):
        fresh._handle_resume(path, 1, 0)
