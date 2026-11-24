# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 action-stage recipe over the generic WorldModel orchestration.

The wrapper owns VAE/CLIP/text encoding and channel-concat conditioning. In
particular, zero-padded clean video latents are only a shape template for noise;
Libero does NOT replace the first noisy latent with a clean latent by default.
The draw order, the 1000-step training timestep, the bsmntw weight and the
per-token action reduction are L3/L4 declarations of the VPP2 provider
(``RngPolicy``, ``RectifiedFlow(num_train_timesteps=1000)``, ``Bsmntw``,
``TokenMeanSquaredError``); the generic ``WorldModel`` executes them. This
recipe adds only VPP2's own policies: the frozen pure-noise video stream, the
terminal-padding exception and the optional first-frame fusion.
``VPP2ImageEncoder`` is the wrapper's CLIP tower with master's dtype rules.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from ..dreamzero.modules.wan_video_image_encoder import WanImageEncoder
from .builder import DiffusionDraw, DiffusionRuntime, WorldModel
from .diffusion.loss import Bsmntw, TokenMeanSquaredError
from .diffusion.timesteps import Constant, RngPolicy, UniformShift
from .layout import TokenLayout
from .spec.world import WorldModelSpec
from .streams.conditions import ConditionBus
from .streams.engine import EngineOutput, JointAttentionEngine, StreamEngine

__all__ = ["VPP2ImageEncoder", "VPP2TrainingWorldModel"]

# Master ``LiberoActionModel.training_loss`` draws video noise, then (no draw)
# the pure-noise video t, then action noise and the action t, and casts t to
# the sample dtype before noising (``scheduler.py:sample_training_t``).
_MASTER_RNG = RngPolicy(order="noise_first", sigma_dtype="like_sample")


class VPP2TrainingWorldModel(WorldModel):
    """Action-only VPP2; forward is inherited without replacing the architecture.

    Batch: stream ``batch_key`` tensors video [B,C,F,H,W], action [B,T,D],
    plus each condition's ``source_key``. Optional masks: action_is_pad [B,T],
    action_terminal_mask [B,T] (subset of padding), image_is_pad [B,N].
    ``fuse_vae_embedding_in_latents`` defaults to False, as in master Libero.
    Explicit True uses the wrapper-materialized video's first clean latent.

    The engine must implement frozen-stream execution. A shallow engine copy
    adds video to frozen_streams without changing an engine shared by callers.
    Shared condition encoders remain trainable through the action branch.
    """

    def __init__(self, spec: WorldModelSpec, *, engine: StreamEngine,
                 layout: TokenLayout, training_stage: str = "action") -> None:
        if training_stage != "action":
            raise NotImplementedError("VPP2TrainingWorldModel supports only training_stage='action'")
        if {stream.name for stream in spec.streams} != {"video", "action"}:
            raise ValueError("VPP2 action training requires video and action streams")
        if len(spec.attention_groups) != 1 or set(spec.attention_groups[0].stream_names) != {"video", "action"}:
            raise ValueError("VPP2 action training requires one video/action attention group")
        sampler = spec.diffusion.sampler_for("action")
        if not isinstance(sampler.distribution, UniformShift) or not sampler.broadcast.is_per_sample:
            raise ValueError("VPP2 action training requires per-sample UniformShift")
        # add_noise trusts these two samplers instead of reading sigma back,
        # which would sync the host every step.
        video_sampler = spec.diffusion.sampler_for("video")
        if (not isinstance(video_sampler.distribution, Constant)
                or video_sampler.distribution.value != 1.0
                or not video_sampler.broadcast.is_per_sample):
            raise ValueError("VPP2 action training requires pure-noise video, per-sample Constant(1.0)")
        # Master parity rests on these provider declarations; refuse to train a
        # recipe whose numbers silently differ from master.
        if list(spec.diffusion.timesteps) != ["video", "action"] or any(
                spec.diffusion.sampler_for(name).rng != _MASTER_RNG for name in ("video", "action")):
            raise ValueError("VPP2 action training draws video then action, noise first, "
                             f"with t cast to the sample dtype ({_MASTER_RNG})")
        if spec.diffusion.schedule.num_train_timesteps != 1000:
            raise ValueError("VPP2 action training uses the 1000-step training timestep")
        if not isinstance(engine, JointAttentionEngine):
            raise TypeError("VPP2 action training requires JointAttentionEngine frozen_streams support")
        if "action" in engine.frozen_streams:
            raise ValueError("the action stream must not be frozen")
        recipe_engine = copy(engine)
        recipe_engine.frozen_streams = engine.frozen_streams | frozenset({"video"})
        super().__init__(spec, engine=recipe_engine, layout=layout)
        self.training_stage = training_stage
        self.blocks["video"].requires_grad_(False)
        for head in spec.heads:
            if head.source_stream == "video":
                self.heads[head.name].requires_grad_(False)
        for condition in spec.conditions:
            if condition.name in self.condition_encoders and set(condition.targets) == {"video"}:
                self.condition_encoders[condition.name].requires_grad_(False)
        action_losses = [loss for loss in spec.diffusion.losses if loss.head == "action_flow"]
        if len(action_losses) != 1:
            raise ValueError("VPP2 action training requires exactly one action_flow loss")
        if (not isinstance(action_losses[0].objective, TokenMeanSquaredError)
                or not isinstance(action_losses[0].weight_of, Bsmntw)):
            raise ValueError("VPP2 action_flow loss is TokenMeanSquaredError weighted by Bsmntw")
        self._action_loss_spec = action_losses[0]

    def _latents(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        streams = {stream.name: stream for stream in self.spec.streams}
        video = self._raw_tensor(batch, streams["video"])
        action = self._raw_tensor(batch, streams["action"])
        if video.ndim != 5 or action.ndim != 3 or video.shape[0] != action.shape[0]:
            raise ValueError("video [B,C,F,H,W] and action [B,T,D] must share batch size")
        if not video.is_floating_point() or not action.is_floating_point():
            raise ValueError("video and action must be floating point")
        if not all(video.shape) or not all(action.shape):
            raise ValueError("video and action dimensions must be nonempty")
        return video, action

    def build_conditions(self, batch: Mapping[str, Any]) -> ConditionBus:
        return super().build_conditions(batch)

    def add_noise(self, batch: Mapping[str, Any],
                  draw: DiffusionDraw) -> DiffusionRuntime:
        video, action = self._latents(batch)
        # Values are not read back: __init__ pins the samplers (video
        # Constant(1.0), action UniformShift on [0, 1]).
        for name, tensor in (("video", video), ("action", action)):
            if draw.sigma[name].shape != (tensor.shape[0],):
                raise ValueError(f"{name} timestep must be [B]")
        raw_pad = batch.get("action_is_pad")
        if raw_pad is not None:
            if not isinstance(raw_pad, Tensor) or raw_pad.shape != action.shape[:2]:
                raise ValueError("action_is_pad must match action [B,T]")
            raw_pad = raw_pad.to(device=action.device, dtype=torch.bool).clone()
        loss_pad = raw_pad
        terminal = batch.get("action_terminal_mask")
        if terminal is not None:
            if raw_pad is None or not isinstance(terminal, Tensor) or terminal.shape != raw_pad.shape:
                raise ValueError("action_terminal_mask must match action_is_pad [B,T]")
            terminal = terminal.to(device=action.device, dtype=torch.bool)
            # Not checked to lie inside the padding, which would sync: the
            # dataset builds it as action_is_pad & tail, and a terminal step
            # outside the padding counts in the loss either way.
            loss_pad = raw_pad & ~terminal
        image_pad = batch.get("image_is_pad")
        if image_pad is not None and (
                not isinstance(image_pad, Tensor) or image_pad.ndim != 2
                or image_pad.shape[0] != video.shape[0]):
            raise ValueError("image_is_pad must be [B,N]")
        runtime = super().add_noise(batch, draw)
        # sigma=1: the video input is the noise itself, as in master.
        noised_video = draw.noise["video"]
        if batch.get("fuse_vae_embedding_in_latents", False):
            noised_video = noised_video.clone()
            noised_video[:, :, :1] = video[:, :, :1].detach()
        runtime.noised["video"] = noised_video
        if loss_pad is not None:
            # Terminal steps are padded in the data but still supervised.
            runtime.loss_valid["action"] = ~loss_pad
        return runtime

    def stream_inputs(self, runtime: DiffusionRuntime) -> dict[str, Tensor]:
        return super().stream_inputs(runtime)

    def compute_losses(self, output: EngineOutput, runtime: DiffusionRuntime,
                       outputs: Mapping[str, Tensor], *,
                       step: int) -> tuple[Tensor, dict[str, Any]]:
        del output
        loss_action = (self._loss_value(self._action_loss_spec, runtime, outputs)
                       * self._action_loss_spec.weight_at(step))
        return loss_action, {"loss_action": loss_action.detach(),
                             "loss_video": loss_action.detach().new_zeros(())}


class VPP2ImageEncoder(WanImageEncoder):
    """The Wan CLIP tower with master VPP2's dtype rules; DreamZero keeps its own.

    Master's ``WanImageEncoderCLIP`` (``model/vpp2/modules/wan_video_clip.py``)
    and the dreamzero ``WanImageEncoder`` run the same tower and weights but
    differ in two places, which made every arch CLIP feature differ from
    master's. Master casts the frames to fp16 before the bicubic resize and
    feeds the resized pixels to the tower as they are. It also adds the fp16
    cls/pos embeddings without casting them, so under bf16 autocast
    ``torch.cat`` promotes the residual stream to fp32.
    """

    def encode_image(self, videos):
        """``[1, 3, H, W]`` frames per sample -> the 31-block features, as master."""
        size = (self.model.image_size,) * 2
        frames = torch.cat([F.interpolate(u.to(torch.float16), size=size, mode="bicubic",
                                          align_corners=False) for u in videos])
        frames = self.transforms.transforms[-1](frames.mul(0.5).add(0.5))
        visual = self.model.visual
        x = visual.patch_embedding(frames).flatten(2).permute(0, 2, 1)
        x = torch.cat([visual.cls_embedding.expand(x.size(0), -1, -1), x], dim=1)
        x = visual.dropout(x + visual.pos_embedding)
        if visual.pre_norm is not None:
            x = visual.pre_norm(x)
        return visual.transformer[:-1](x)
