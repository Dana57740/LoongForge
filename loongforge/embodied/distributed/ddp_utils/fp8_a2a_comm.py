# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FP8 AllToAll + FP8 AllGather DDP gradient communication.

Replaces DDP's bf16 ring AllReduce with

    quantize -> AllToAll(fp8) -> local fp32 reduce -> quantize -> AllGather(fp8)
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - guarded by validate_runtime().
    triton = None
    tl = None

logger = logging.getLogger(__name__)

DEFAULT_BLOCK = 256
# Buckets below this size fall back to plain AllReduce: two collectives plus four
# kernels do not pay for themselves on a few MiB, and AllToAll efficiency drops
# off below roughly 64 MiB per rank.
DEFAULT_MIN_MIB = 8.0
# Total resident scratch across all buckets. Buckets that do not fit degrade to
# full precision; see _scratch_for.
DEFAULT_MAX_SCRATCH_GB = 24.0
# Quantization blocks handled per Triton program. Decouples the tile size from
# the quantization block size: a 220 MB bucket at BLOCK=256 would otherwise
# need ~430k programs each touching only 256 elements.
NUM_BLOCKS_PER_TILE = 8
# Upper bound on ``block``: a program materializes NUM_BLOCKS_PER_TILE * block
# fp32 values, so 1024 already costs 32 KiB of registers per program.
MAX_BLOCK = 1024

_BLOCK = DEFAULT_BLOCK
_MIN_BYTES = int(DEFAULT_MIN_MIB * 2**20)
_MAX_SCRATCH_BYTES = int(DEFAULT_MAX_SCRATCH_GB * 2**30)


FLAG = "--ddp-comm-hook fp8_a2a_allgather_hook"


def validate_runtime(device, backend: str) -> None:
    """Fail at install time when the FP8 AllToAll kernels cannot run.

    Called once from ``parallel.py`` before the hook is registered. Checking
    here rather than inside the hook means a CPU run, a gloo run, or a pre-Ada
    GPU fails before the model and dataloader are built, instead of surfacing as
    a Triton compile error from inside the DDP reducer at the first backward.
    """
    device = torch.device(device)
    entries = [e.strip() for e in str(backend).lower().split(",") if e.strip()]
    scoped = dict(e.rsplit(":", 1) for e in entries if ":" in e)
    plain = next((e for e in entries if ":" not in e), "")
    resolved = scoped.get(device.type, plain)
    context = f"device={device}, backend={str(backend).lower()}"

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{FLAG} requires a CUDA device; got {context}")
    if resolved != "nccl":
        raise RuntimeError(
            f"{FLAG} requires the NCCL backend; got {context} "
            f"(resolved {device.type}:{resolved or 'unknown'})"
        )
    if triton is None or tl is None or not hasattr(tl, "float8e4nv"):
        raise RuntimeError(
            f"{FLAG} requires Triton with the FP8 type tl.float8e4nv; got {context}"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(f"{FLAG} requires PyTorch FP8 E4M3 support; got {context}")
    # tl.float8e4nv needs Ada or newer. On cc 8.0 the import and the type both
    # look fine, and only Triton's first compile -- i.e. the first backward --
    # fails, which is the late failure this function exists to move forward.
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        raise RuntimeError(
            f"{FLAG} requires compute capability >= 8.9 for tl.float8e4nv; "
            f"got {capability} ({context})"
        )
    try:
        target = triton.runtime.driver.active.get_current_target()
        triton_backend = str(target.backend).lower()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize Triton's CUDA backend for {FLAG} ({context})"
        ) from exc
    if triton_backend != "cuda":
        raise RuntimeError(
            f"{FLAG} requires Triton's CUDA backend; "
            f"got triton_backend={triton_backend!r} ({context})"
        )


if triton is not None:

    @triton.jit
    def _quantize_fused_kernel(
        X,                      # input, num_chunks * S elements
        OUT_U8,                 # fused uint8 buffer
        OUT_F32,                # same storage viewed as fp32
        numel,                  # valid elements in X; the rest is padding
        S,                      # elements per chunk
        chunk_u8,               # bytes per chunk = S + 4 * S // BLOCK
        chunk_f32,              # chunk_u8 // 4
        scale_base_f32,         # S // 4, fp32-offset of the scale region
        tiles_per_chunk,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Blockwise-quantize ``X`` into ``num_chunks`` fused fp8+scale chunks.

        ``X`` is read straight out of ``bucket.buffer()``. The padding needed to
        make ``num_chunks * S`` a clean multiple is handled by masking the load,
        not by materializing a padded copy -- that copy would cost another full
        bucket of scratch (and HBM traffic) per bucket.

        Offsets are int64: DDP hands us buckets of up to 6.0e9 elements, and the
        fused buffer is 8x that in bytes, so int32 indexing silently wraps and
        lands as cudaErrorIllegalAddress.
        """
        pid = tl.program_id(0).to(tl.int64)
        chunk = pid // tiles_per_chunk
        blk0 = (pid % tiles_per_chunk) * NB

        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]

        idx = chunk * S + within
        x = tl.load(X + idx, mask=idx < numel, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        scale = amax / 448.0  # E4M3_MAX; Triton cannot read non-constexpr globals
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = (x * inv[:, None]).to(tl.float8e4nv)

        tl.store(OUT_U8 + chunk * chunk_u8 + within, q.to(tl.uint8, bitcast=True))
        tl.store(OUT_F32 + chunk * chunk_f32 + scale_base_f32 + blk0 + ob, scale)

    @triton.jit
    def _dequant_reduce_kernel(
        RECV_U8,
        RECV_F32,
        OUT,                    # averaged shard, S elements
        chunk_u8,
        chunk_f32,
        scale_base_f32,
        world_size,
        inv_world,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Sum all ranks' fp8 chunks in fp32 and write the averaged shard.

        This is where DDP's ``/world_size`` happens, in fp32, exactly once.
        """
        cu = chunk_u8.to(tl.int64)
        cf = chunk_f32.to(tl.int64)
        blk0 = tl.program_id(0).to(tl.int64) * NB
        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]
        s_within = blk0 + ob

        acc = tl.zeros((NB, BLOCK), dtype=tl.float32)
        for r in range(world_size):
            q = tl.load(RECV_U8 + r * cu + within)
            d = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc = tl.load(RECV_F32 + r * cf + scale_base_f32 + s_within)
            acc += d * sc[:, None]

        tl.store(OUT + within, acc * inv_world)

    @triton.jit
    def _dequant_scatter_kernel(
        AG_U8,
        AG_F32,
        OUT,                    # full gradient bucket, written in place
        numel,                  # valid elements in OUT
        S,
        chunk_u8,
        chunk_f32,
        scale_base_f32,
        tiles_per_chunk,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Dequantize each rank's gathered shard into its slice of the bucket."""
        pid = tl.program_id(0).to(tl.int64)
        chunk = pid // tiles_per_chunk
        blk0 = (pid % tiles_per_chunk) * NB

        ob = tl.arange(0, NB).to(tl.int64)
        oe = tl.arange(0, BLOCK).to(tl.int64)
        within = (blk0 + ob)[:, None] * BLOCK + oe[None, :]

        q = tl.load(AG_U8 + chunk * chunk_u8 + within)
        d = q.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sc = tl.load(AG_F32 + chunk * chunk_f32 + scale_base_f32 + blk0 + ob)
        idx = chunk * S + within
        tl.store(OUT + idx, d * sc[:, None], mask=idx < numel)


class _BucketScratch:
    """Persistent per-bucket scratch, keyed by bucket identity.

    Both collectives move one fused ``uint8`` buffer of ``world_size`` equal
    chunks::

        chunk_j = [ S bytes fp8 payload ][ 4 * S/block bytes fp32 block scales ]

    Fusing the scales into the payload keeps this at one collective per hop
    instead of two. ``S`` is a multiple of ``block * NUM_BLOCKS_PER_TILE``, so
    the fp32 region is 4-byte aligned and the buffer can be viewed as fp32.

    Allocated once and never freed. That is deliberate: allocating comm scratch
    inside the hook every step is what makes the caching allocator hand out
    storage that NCCL is still reading, which shows up as an OOM/FATAL a few
    steps in rather than as a clean error.

    Cost per bucket is ``2 * padded_bytes * 1.0156 + shard_bytes``, i.e. about
    ``1.15x`` the bucket itself. Buffers are aliased where the dataflow allows:
    ``send`` is reused as the AllGather input once AllToAll has consumed it, and
    ``recv`` is reused as the AllGather output once the local reduce has drained
    it.
    """

    __slots__ = ("send", "recv", "shard", "S", "chunk_u8", "tiles_per_chunk",
                 "numel", "identity")

    def __init__(self, identity, numel: int, dtype, device, world_size, block):
        self.S, self.chunk_u8, _ = self.plan(numel, dtype, world_size, block)
        self.numel = numel
        self.identity = identity
        self.tiles_per_chunk = self.S // (block * NUM_BLOCKS_PER_TILE)

        total = world_size * self.chunk_u8
        self.send = torch.empty(total, dtype=torch.uint8, device=device)
        self.recv = torch.empty(total, dtype=torch.uint8, device=device)
        self.shard = torch.empty(self.S, dtype=dtype, device=device)

    @staticmethod
    def plan(numel: int, dtype, world_size: int, block: int):
        """Size the scratch without allocating it.

        Kept as the single source of truth for the layout so the budget check
        can run *before* allocation. Sizing after allocating cannot prevent the
        OOM it exists to prevent.
        """
        align = block * NUM_BLOCKS_PER_TILE
        # Elements per chunk, rounded up so every chunk is a whole number of
        # tiles. Wasted elements are at most world_size * align - 1.
        per_rank = (numel + world_size - 1) // world_size
        S = (per_rank + align - 1) // align * align
        chunk_u8 = S + 4 * (S // block)
        return S, chunk_u8, 2 * world_size * chunk_u8 + S * dtype.itemsize

    def bytes(self) -> int:
        return self.send.numel() + self.recv.numel() + self.shard.nbytes


_SCRATCH: dict[int, _BucketScratch] = {}
_SCRATCH_BYTES = 0
# Buckets already reported as over budget, so the warning fires once per bucket
# instead of once per step.
_OVER_BUDGET: set[int] = set()


def _scratch_for(index, identity, numel, dtype, device, world_size, block, budget):
    """Get or (re)allocate scratch for one bucket, keyed by bucket index.

    Keyed by ``bucket.index()`` rather than by full identity so that DDP's
    bucket rebuild after the first iteration *replaces* the old scratch instead
    of accumulating a second full generation of it (~14 GiB each at
    bucket_cap_mb=200).

    Rebuild is otherwise harmless here: these are pure scratch buffers with no
    state carried across steps, so a changed parameter-to-offset mapping cannot
    silently corrupt anything.
    """
    global _SCRATCH_BYTES
    scratch = _SCRATCH.get(index)
    if scratch is not None:
        if scratch.identity == identity:
            return scratch
        _SCRATCH_BYTES -= scratch.bytes()
        del _SCRATCH[index]
        del scratch
        logger.info("fp8_a2a: bucket %d layout changed, reallocating scratch", index)

    need = _BucketScratch.plan(numel, dtype, world_size, block)[2]
    if _SCRATCH_BYTES + need > budget:
        # Degrade, do not abort. A bucket layout we cannot afford is a reason to
        # send this bucket at full precision, not to kill the training job: the
        # hook is an optimisation and every caller has a correct fallback. DDP's
        # iteration-0 bucket alone is 12 GiB of bf16 on FastWAM, and it appears
        # exactly once, so passing it through costs nothing in steady state.
        if index not in _OVER_BUDGET:
            _OVER_BUDGET.add(index)
            logger.warning(
                "fp8_a2a: bucket %d needs %.2f GiB scratch, budget %.2f GiB "
                "with %.2f GiB already held -- falling back to full-precision "
                "AllReduce for this bucket. Raise --ddp-comm-hook-fp8-max-scratch-gb "
                "or use a larger ddp_bucket_cap_mb to quantize it.",
                index, need / 2**30, budget / 2**30, _SCRATCH_BYTES / 2**30,
            )
        return None

    scratch = _BucketScratch(identity, numel, dtype, device, world_size, block)
    _SCRATCH[index] = scratch
    _SCRATCH_BYTES += scratch.bytes()
    logger.info(
        "fp8_a2a: bucket %d numel=%d S=%d chunk_u8=%d scratch=%.1f MiB "
        "(total %.2f GiB over %d buckets)",
        index, numel, scratch.S, scratch.chunk_u8,
        scratch.bytes() / 2**20, _SCRATCH_BYTES / 2**30, len(_SCRATCH),
    )
    return scratch


def reset_scratch() -> None:
    """Drop all cached scratch. For tests and for bucket-layout changes."""
    global _SCRATCH_BYTES
    _SCRATCH.clear()
    _SCRATCH_BYTES = 0
    _OVER_BUDGET.clear()


def quantize_chunks(x, out_u8, numel, S, chunk_u8, num_chunks, block):
    """Quantize ``x`` into ``num_chunks`` fused fp8+scale chunks of ``out_u8``."""
    tiles_per_chunk = S // (block * NUM_BLOCKS_PER_TILE)
    _quantize_fused_kernel[(num_chunks * tiles_per_chunk,)](
        x, out_u8, out_u8.view(torch.float32),
        numel, S, chunk_u8, chunk_u8 // 4, S // 4, tiles_per_chunk,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def dequant_reduce(recv_u8, out, S, chunk_u8, world_size, block):
    """Sum every rank's fp8 chunk in fp32 and write the averaged shard."""
    tiles = S // (block * NUM_BLOCKS_PER_TILE)
    _dequant_reduce_kernel[(tiles,)](
        recv_u8, recv_u8.view(torch.float32), out,
        chunk_u8, chunk_u8 // 4, S // 4, world_size, 1.0 / world_size,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def dequant_scatter(ag_u8, out, numel, S, chunk_u8, num_chunks, block):
    """Dequantize gathered shards back into the gradient bucket, in place."""
    tiles_per_chunk = S // (block * NUM_BLOCKS_PER_TILE)
    _dequant_scatter_kernel[(num_chunks * tiles_per_chunk,)](
        ag_u8, ag_u8.view(torch.float32), out,
        numel, S, chunk_u8, chunk_u8 // 4, S // 4, tiles_per_chunk,
        BLOCK=block, NB=NUM_BLOCKS_PER_TILE,
    )


def configure(block: int = DEFAULT_BLOCK, min_mib: float = DEFAULT_MIN_MIB,
              max_scratch_gb: float = DEFAULT_MAX_SCRATCH_GB) -> None:
    """Set the hook's tunables. Called once from ``parallel.py`` at install time.

    DDP fixes the comm-hook signature at ``(state, bucket)``, so the knobs cannot
    be passed per call and have to live in module state.

    Changing ``block`` changes the wire layout, so any scratch allocated under
    the previous value is dropped rather than silently reused with a stale
    ``chunk_u8``.
    """
    global _BLOCK, _MIN_BYTES, _MAX_SCRATCH_BYTES
    # Power of two because all three kernels index with tl.arange(0, BLOCK).
    if block <= 0 or block & (block - 1):
        raise ValueError(
            f"ddp_comm_hook_fp8_block must be a positive power of two, got {block}"
        )
    # Each program holds an NUM_BLOCKS_PER_TILE x block fp32 tile in registers,
    # so the usable ceiling is far below Triton's own tl.arange limit.
    if block > MAX_BLOCK:
        raise ValueError(
            f"ddp_comm_hook_fp8_block must be <= {MAX_BLOCK}, got {block}"
        )
    if min_mib < 0:
        raise ValueError(
            f"ddp_comm_hook_fp8_min_mib must be >= 0, got {min_mib}"
        )
    if max_scratch_gb < 0:
        raise ValueError(
            f"ddp_comm_hook_fp8_max_scratch_gb must be >= 0, got {max_scratch_gb}"
        )
    if block != _BLOCK:
        reset_scratch()
    _BLOCK = block
    _MIN_BYTES = int(min_mib * 2**20)
    _MAX_SCRATCH_BYTES = int(max_scratch_gb * 2**30)
    logger.info(
        "fp8_a2a: block=%d min_mib=%g max_scratch_gb=%g", block, min_mib, max_scratch_gb
    )


def _config():
    return _BLOCK, _MIN_BYTES, _MAX_SCRATCH_BYTES


def fp8_a2a_allgather_hook(process_group, bucket):
    """DDP comm hook: fp8 AllToAll + local fp32 reduce + fp8 AllGather.

    Egress is ``0.508x`` of ring AllReduce. Every element is quantized once per
    half, so the error is comparable to a single fp8 round-trip rather than
    compounding across ring hops.

    Small buckets fall back to plain AllReduce: two collectives plus four
    kernels do not pay for themselves when the payload is only a few MiB, and
    the AllToAll efficiency drops off below ~64 MiB per rank. Buckets whose
    scratch does not fit the budget take the same fallback.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    tensor = bucket.buffer()
    block, min_bytes, budget = _config()

    if world_size < 2 or tensor.nbytes < min_bytes:
        return allreduce_hook(process_group, bucket)

    identity = (tensor.numel(), tensor.dtype, tensor.device, id(group), block)
    st = _scratch_for(
        bucket.index(), identity, tensor.numel(), tensor.dtype, tensor.device,
        world_size, block, budget,
    )
    if st is None:
        return allreduce_hook(process_group, bucket)
    S, chunk_u8 = st.S, st.chunk_u8

    quantize_chunks(tensor, st.send, st.numel, S, chunk_u8, world_size, block)
    a2a = dist.all_to_all_single(st.recv, st.send, group=group, async_op=True)

    result = torch.futures.Future()

    def after_a2a(fut):
        try:
            fut.wait()
            # fp32 accumulate across ranks, divide by world_size, requantize.
            dequant_reduce(st.recv, st.shard, S, chunk_u8, world_size, block)
            ag_send = st.send[:chunk_u8]
            quantize_chunks(st.shard, ag_send, S, S, chunk_u8, 1, block)
            work = dist.all_gather_into_tensor(
                st.recv, ag_send, group=group, async_op=True
            )
        except Exception as exc:
            result.set_exception(exc)
            return

        def after_ag(fut):
            try:
                fut.wait()
                dequant_scatter(
                    st.recv, tensor, st.numel, S, chunk_u8, world_size, block
                )
                result.set_result(tensor)
            except Exception as exc:
                result.set_exception(exc)

        work.get_future().then(after_ag)

    a2a.get_future().then(after_a2a)
    return result




