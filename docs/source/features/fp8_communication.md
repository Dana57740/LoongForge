# FP8 Communication

LoongForge provides two opt-in FP8 communication optimizations for the embodied training stack. Both reduce cross-rank traffic by communicating in FP8 E4M3 instead of BF16/FP32, and both change **communication precision only** — forward and backward computation are unaffected, and neither converts weights, activations, or GEMMs to FP8 (that is the separate end-to-end [FP8 training](fp8_training.md) feature).

Pick the one that matches your parallel strategy; they operate on different collectives and are not used together.

| | [FSDP2 Delta-FP8 Param AllGather](delta_fp8_allgather.md) | [DDP FP8 Grad AllReduce](fp8_grad_comm_hook.md) |
| --- | --- | --- |
| Parallel strategy | FSDP2 | DDP |
| Compresses | Parameter AllGather | Gradient all-reduce |
| Method | Blockwise FP8 delta vs a persistent BF16 reference | FP8 AllToAll → local FP32 reduce → FP8 AllGather |
| Enable with | `--fsdp-delta-fp8-allgather` | `--ddp-comm-hook fp8_a2a_allgather_hook` |

Support in the communication path does not by itself establish numerical or performance suitability for every model. Validate loss, step time, and peak memory against the same model's BF16/FP32 baseline before adopting either in a new recipe.

```{toctree}
:maxdepth: 1

delta_fp8_allgather
fp8_grad_comm_hook
```
