# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The gate: ``--model-name`` → YAML → typed config → ``ArchWorldModel``.

Everything the other ``test_arch_*`` files check lives inside the ``arch``
package. This file checks the four seams that connect it to the trainer, each of
which fails silently rather than loudly when it breaks:

1. ``MODEL_SCHEMA`` rows resolve to YAML files that exist and merge cleanly into
   the typed config classes.
2. ``geometry.num_frames`` equals the sampled RGB frame count. The VAE sees the
   frames the dataset sampled, so these two numbers are one fact written twice;
   a disagreement only surfaces as a latent-grid shape error deep in a forward.
3. The model registry maps both arch names onto the one shell, and does *not*
   steal ``fastwam`` from ``FastWAMPolicy``.
4. The shell really converts a pixel batch into ``(loss, metrics)`` — with stub
   encoders, because the real VAE allocates its normalisation buffers on CUDA.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from loongforge.embodied.model.arch.builder import WorldModel
from loongforge.embodied.model.arch.diffusion.loss import WeightRamp
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.modeling_arch import ArchWorldModel
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.registry import MODEL_REGISTRY
from loongforge.embodied.train.config_map import get_config_path, get_model_schema
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer

from test_arch_blocks import BATCH, CONTEXT_LEN, tiny_spec

ARCH_MODELS = ("vpp2", "internw0_delta")


def load_config(model_name: str):
    """Reproduce the parser's merge for one model name.

    Mirrors ``train/parser.py``: a structured default built from the schema's
    config class, with the YAML ``model:`` section merged on top.

    Args:
        model_name: A ``MODEL_SCHEMA`` key.

    Returns:
        The materialised model config instance.
    """
    schema = get_model_schema(model_name)
    yaml_cfg = OmegaConf.load(get_config_path(model_name))
    merged = OmegaConf.merge(OmegaConf.structured(schema.model_config_cls),
                             yaml_cfg.get("model", {}))
    return OmegaConf.to_object(merged)


# ---- 1. schema rows and the YAML merge ----


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_yaml_merges_into_the_typed_config(model_name):
    """The YAML is a set of overrides on the dataclass, not a free-form dict.

    ``OmegaConf.to_object`` on a structured merge rejects unknown keys and wrong
    types, so this one call covers every field name in the file.
    """
    cfg = load_config(model_name)
    assert cfg.model_type == model_name
    assert cfg.build_spec().model_type == model_name


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_model_identity_is_not_written_in_the_yaml(model_name):
    """Rule R5: no YAML line names the model, a class path or a topology.

    ``model_type`` comes from the config subclass default, which the
    ``--model-name`` argument selected through ``MODEL_SCHEMA``. A file that also
    wrote it would let the two disagree, and the loser would be silent.
    """
    section = OmegaConf.load(get_config_path(model_name)).model
    assert "model_type" not in section
    assert "_target_" not in section
    # Trainable DiT weights arrive via ``--pretrained-checkpoint``; ``model_id``
    # is the encoder stack and is a different fact.
    assert "pretrained" not in section


# ---- 2. the frame count the dataset and the VAE have to agree on ----


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_geometry_frame_count_matches_the_dataloader(model_name):
    """``geometry.num_frames`` is the pixel frame count handed to the VAE.

    VPP2 samples observation offsets from its raw ``num_frames`` window;
    InternW0-Delta uses FastWAM's explicit ``num_video_frames`` count.
    Either count must agree with the geometry before a real VAE forward.
    """
    yaml_cfg = OmegaConf.load(get_config_path(model_name))
    if model_name == "vpp2":
        from loongforge.embodied.data.datasets.vpp2.transforms import VPP2DataConfig

        data_cfg = OmegaConf.to_object(OmegaConf.merge(
            OmegaConf.structured(VPP2DataConfig), yaml_cfg.data))
        sampled_frames = len(data_cfg.observation_delta_indices)
    else:
        sampled_frames = yaml_cfg.data.num_video_frames
    assert yaml_cfg.model.geometry.num_frames == sampled_frames
    assert (yaml_cfg.model.geometry.action_video_freq_ratio
            == yaml_cfg.data.action_video_freq_ratio)


# ---- 3. the registries ----


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_gate_routes_both_arch_names_to_one_shell(model_name):
    """Both arch models are the same shell; what differs is the spec it holds."""
    assert MODEL_REGISTRY[model_name] is ArchWorldModel


def test_gate_does_not_take_over_fastwam():
    """``fastwam`` keeps its own policy class.

    ``@register_model`` decorators stack, so claiming one name too many is a
    one-character mistake that would silently reroute an existing model.
    """
    from loongforge.embodied.model.fastwam.modeling_fastwam import FastWAMPolicy

    assert MODEL_REGISTRY["fastwam"] is FastWAMPolicy


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_arch_models_route_to_their_data_preprocessor(model_name):
    """VPP2 uses its native data contract; InternW0-Delta keeps FastWAM's."""
    import loongforge.embodied.data.datasets.fastwam.transforms  # noqa: F401
    from loongforge.embodied.data.datasets.fastwam.transforms.fastwam_collator \
        import FastWAMPreprocessor
    from loongforge.embodied.data.datasets.transforms.collator import (
        _PREPROCESSOR_REGISTRY)
    from loongforge.embodied.data.datasets.vpp2.transforms import VPP2Preprocessor

    expected = VPP2Preprocessor if model_name == "vpp2" else FastWAMPreprocessor
    assert _PREPROCESSOR_REGISTRY[model_name] is expected


@pytest.mark.parametrize("model_name", ARCH_MODELS)
def test_config_exposes_what_the_collator_reads(model_name):
    """The four attributes ``FastWAMPreprocessor`` reads off ``model_cfg``.

    Two of them are properties read back off the built spec rather than fields:
    restating the action horizon or the text width in YAML would let the data
    pipeline and the model disagree by one edit.
    """
    cfg = load_config(model_name)
    assert cfg.action_horizon == cfg.build_spec().geometry.action_horizon
    assert cfg.video_dit_config["text_dim"] == cfg.build_spec().condition("text").width
    assert cfg.tokenizer_max_len > 0
    assert cfg.model_id


# ---- 4. the shell, end to end on stub encoders ----


class StubVAE(torch.nn.Module):
    """Stand-in for the Wan VAE: pixels in, a latent grid of the right shape out.

    The real encoder allocates its mean/std buffers on CUDA at construction
    time, which is why ``ArchWorldModel`` takes both encoders by injection.
    """

    def __init__(self, spec: WorldModelSpec) -> None:
        super().__init__()
        self.channels = spec.geometry.latent_channels
        self.grid = spec.geometry.latent_grid

    def encode(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Return ``[B, C, F, H, W]`` derived from the pixel batch size only."""
        return torch.randn(video.shape[0], self.channels, *self.grid)


class StubImageEncoder(torch.nn.Module):
    """Stand-in for the CLIP image encoder: one frame in, a token grid out."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def encode_image(self, frames: torch.Tensor) -> torch.Tensor:
        """Return ``[B, CONTEXT_LEN, width]`` for a ``[B, 1, 3, H, W]`` input."""
        assert frames.ndim == 5 and frames.shape[1] == 1
        return torch.randn(frames.shape[0], CONTEXT_LEN, self.width)


def make_shell(spec: WorldModelSpec) -> ArchWorldModel:
    """A shell around ``spec``, wired to stub encoders.

    The core is the generic ``WorldModel``, so the config names the stage that
    path serves (whole-video VAE encode). ``from_pretrained`` refuses that stage
    for VPP2 training; the batch plumbing tested here is shared with it.
    """
    from loongforge.embodied.model.arch.model_configuration_arch import Vpp2Config

    core = WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=TokenLayout.from_geometry(spec.geometry,
                                                       spec.streams))
    return ArchWorldModel(
        Vpp2Config(dtype="float32", training_stage="joint"), core,
        vae=StubVAE(spec),
        image_encoder=StubImageEncoder(spec.condition("clip_image").width))


@pytest.fixture
def shell() -> ArchWorldModel:
    """A vpp2 shell at test width, wired to stub encoders.

    The spec comes from the real provider (so the condition table, visibility
    and loss terms are the production ones) with the widths and geometry shrunk
    — the shell's job is tensor plumbing, which is width-independent.
    """
    return make_shell(tiny_spec("vpp2"))


def pixel_batch(shell: ArchWorldModel) -> dict:
    """A FastWAM-style sample dict: pixels, actions, cached text, proprio.

    This is exactly what ``FastWAMPreparedBatch.to_sample()`` produces, which is
    the point — the shell's input contract is the collator's output contract.
    """
    geo = shell.spec.geometry
    text = shell.spec.condition("text")
    return {
        "video": torch.rand(BATCH, 3, geo.num_frames, geo.frame_height,
                            geo.frame_width * geo.num_views) * 2 - 1,
        "action": torch.randn(BATCH, geo.action_horizon, geo.action_dim),
        "context": torch.randn(BATCH, CONTEXT_LEN, text.width),
        "context_mask": torch.ones(BATCH, CONTEXT_LEN, dtype=torch.bool),
        "proprio": torch.randn(BATCH, geo.action_horizon,
                               shell.spec.condition("proprio").source_width),
    }


def test_shell_turns_a_pixel_batch_into_a_scalar_loss(shell):
    """The trainer-facing contract: a pixel batch in, ``(loss, metrics)`` out."""
    loss, metrics = shell(pixel_batch(shell))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for term in shell.spec.diffusion.losses:
        assert metrics[f"loss_{term.name}"].ndim == 0


def test_shell_fills_one_batch_key_per_stream_and_condition(shell):
    """``encode_batch`` is driven by the spec, not by a per-model key list.

    Stream keys come from ``StreamSpec.batch_key`` and condition keys from
    ``ConditionSpec.source_key``, so adding a condition to a provider cannot
    leave the shell behind.
    """
    encoded = shell.encode_batch(pixel_batch(shell))
    for stream in shell.spec.streams:
        assert stream.batch_key in encoded
    for cond in shell.spec.conditions:
        if cond.enabled:
            assert cond.source_key in encoded


def test_shell_checks_the_latent_grid_against_the_geometry(shell):
    """A VAE that disagrees with the geometry fails here, naming both grids.

    Without this check the mismatch reaches the patch embedding as an
    off-by-one-frame shape error, which reads like a bug in the stream.
    """
    shell.vae.grid = (shell.vae.grid[0] + 1, *shell.vae.grid[1:])
    with pytest.raises(ValueError, match="latent grid"):
        shell(pixel_batch(shell))


def test_i2v_block_is_four_mask_channels_plus_clean_leading_frames(shell):
    """The 20-channel i2v condition: ``mask(4)`` then the clean condition frames.

    The mask is 1 on the conditioned leading frames and the latent half is zero
    everywhere else, which is how "this frame is given" is told apart from "this
    frame happens to be dark".
    """
    cond = shell.spec.condition("cond_latent")
    latents = torch.randn(BATCH, shell.spec.geometry.latent_channels,
                          *shell.spec.geometry.latent_grid)
    block = shell._i2v_channels(cond, latents)
    assert block.shape[1] == cond.width == 4 + latents.shape[1]
    assert torch.all(block[:, :4, :1] == 1.0)
    assert torch.all(block[:, :4, 1:] == 0.0)
    assert torch.equal(block[:, 4:, :1], latents[:, :, :1])
    assert torch.all(block[:, 4:, 1:] == 0.0)


def test_shell_rejects_a_config_of_the_wrong_type():
    """``from_pretrained`` takes a ``WorldModelConfig``, and says so.

    The gate hands over whatever ``MODEL_SCHEMA`` built, so a wrong schema row
    shows up here; the alternative is an ``AttributeError`` on ``build_spec``.
    """
    with pytest.raises(TypeError, match="WorldModelConfig"):
        ArchWorldModel.from_pretrained(object())


def test_vpp2_first_frame_vae_runs_under_the_callers_autocast():
    """As master's ``_encode_video_latents``: fp32 pixels into the VAE, whose
    convolutions the trainer's bf16 autocast runs."""
    from loongforge.embodied.model.arch.model_configuration_arch import Vpp2Config

    spec = tiny_spec("vpp2")
    seen = []

    class FirstFrameVAE(StubVAE):
        def encode(self, video: torch.Tensor, tiled: bool = False) -> torch.Tensor:
            seen.append((video.dtype, torch.is_autocast_enabled(video.device.type)))
            return torch.randn(video.shape[0], self.channels, 1, *self.grid[1:])

    core = WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=TokenLayout.from_geometry(spec.geometry, spec.streams))
    shell = ArchWorldModel(Vpp2Config(dtype="float32"), core, vae=FirstFrameVAE(spec),
                           image_encoder=StubImageEncoder(spec.condition("clip_image").width))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        shell.encode_batch(pixel_batch(shell))
    assert seen == [(torch.float32, True)]


def test_vpp2_joint_stage_is_rejected_on_the_arch_path(monkeypatch):
    """Only the action stage has a recipe checked against master.

    The generic core would accept a joint config and train it without error,
    although it was never compared with master's joint model and leaves the
    action padding unmasked. The check must come before the full-size core is
    built.
    """
    from loongforge.embodied.model.arch import modeling_arch
    from loongforge.embodied.model.arch.model_configuration_arch import Vpp2Config

    def unreachable(**_):
        raise AssertionError("from_pretrained started building a joint VPP2 model")

    monkeypatch.setattr(modeling_arch, "JointAttentionEngine", unreachable)
    with pytest.raises(NotImplementedError, match="training_stage: action"):
        ArchWorldModel.from_pretrained(Vpp2Config(dtype="float32", training_stage="joint"))


def test_trainer_iteration_drives_the_loss_weight_ramp():
    """The trainer's ``iteration`` is the core's ``global_step``.

    ``_select_forward_kwargs`` passes ``iteration`` only to a forward that
    names it, and nothing else writes ``global_step`` into the batch. Without
    it every ``WeightRamp`` would stay at its start value: internw0_delta's
    ``track_align`` (0.0 -> 0.1) and any YAML ``warmup_steps`` would never
    train, so a training forward without it raises.
    """
    spec = tiny_spec("vpp2")
    ramp = WeightRamp(start_step=0, end_step=10, start_value=0.0, end_value=1.0)
    spec = replace(spec, diffusion=replace(spec.diffusion, losses=tuple(
        replace(loss, ramp=ramp) if loss.head == "action_flow" else loss
        for loss in spec.diffusion.losses)))
    shell = make_shell(spec)
    kwargs = FinetuneTrainer._select_forward_kwargs(SimpleNamespace(model=shell), iteration=5)
    assert kwargs == {"iteration": 5}
    _, metrics = shell(pixel_batch(shell), **kwargs)
    assert metrics["action_flow/weight"] == 0.5
    _, metrics = shell(pixel_batch(shell), iteration=10)
    assert metrics["action_flow/weight"] == 1.0
    with pytest.raises(ValueError, match="no global_step"):
        shell(pixel_batch(shell))


def test_batch_global_step_must_agree_with_the_trainer_iteration(shell):
    """Two step counters that disagree raise instead of one winning silently."""
    batch = {**pixel_batch(shell), "global_step": 3}
    shell(batch, iteration=3)
    with pytest.raises(ValueError, match="disagrees"):
        shell(batch, iteration=4)
