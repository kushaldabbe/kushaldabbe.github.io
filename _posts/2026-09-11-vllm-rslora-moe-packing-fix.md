---
title: "First vLLM PR merged: rsLoRA scaling in MoE expert packing"
date: 2026-09-11 13:30:00 +0900
---

My first contribution to [vLLM](https://github.com/vllm-project/vllm) was merged on
11 September 2026: [PR #55548](https://github.com/vllm-project/vllm/pull/55548),
fixing [issue #55545](https://github.com/vllm-project/vllm/issues/55545).

[![vLLM PR #55548 - merged card](https://opengraph.githubassets.com/1/vllm-project/vllm/pull/55548)](https://github.com/vllm-project/vllm/pull/55548)
{: .kd-og-img }

## The bug

`pack_moe` and `pack_moe_stacked` recomputed the LoRA scaling factor as
`lora_alpha / rank`. The dense path (`pack()` + `optimize()`) stores the scaling
factor at pack time, and with `use_rslora=True` that stored value is
`alpha / sqrt(r)` — not `alpha / r`. The MoE path dropped the stored factor and
silently applied a delta `sqrt(r)` times weaker than the dense path.

## The fix

Store the scaling factor during packing and reuse it in the MoE paths, matching
the dense path's behavior. The core change is one line in each MoE packer:
`scaling = lora_alpha / rank` becomes `scaling = first_lora.scaling`. Two files,
+56 −3.

## Verification

`pytest tests/lora/test_lora_weights.py`:

| Suite state | Failed | Passed |
|:---|:---:|:---:|
| Without the fix | 3 | 2 |
| With the fix | 0 | 5 |

The new tests construct adapters with `RSLORA_SCALING = alpha / sqrt(r)` and
assert both MoE packers keep the stored factor — plus the non-gated-MoE case,
where `w3` must stay unscaled. Verified locally on main, CPU-only.
