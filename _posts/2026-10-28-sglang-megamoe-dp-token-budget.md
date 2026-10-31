---
title: "First SGLang PR merged: MegaMoE double-divided the DP prefill budget"
date: 2026-10-28 11:30:00 +0900
---

My first contribution to [SGLang](https://github.com/sgl-project/sglang) merged on
28 October 2026: [PR #46267](https://github.com/sgl-project/sglang/pull/46267),
a fix in the MegaMoE startup path.

[![SGLang PR #46267 - merged card](https://opengraph.githubassets.com/1/sgl-project/sglang/pull/46267)](https://github.com/sgl-project/sglang/pull/46267)
{: .kd-og-img }

## The bug

MegaMoE checks at startup that
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` covers the largest
per-rank prefill batch. With attention DP enabled, that check under-reported the
per-rank token width.

`handle_data_parallelism` already divides `chunked_prefill_size` by
`attn_dp_size` and stores the per-rank value
(`python/sglang/srt/arg_groups/parallel_hook.py:331`). The validator read that
resolved value and divided it a second time
(`python/sglang/srt/arg_groups/mega_moe_hook.py:206-212`).

With `--chunked-prefill-size 16384 --attn-dp-size 2` the real per-rank chunk is
8192, but the validator computed 4096. A cap of 4096 then passed startup, even
though each rank can receive 8192 prefill tokens. At runtime the token-cap
assertion in `mega_moe.py:440` fails during CUDA graph capture. Outside capture,
`should_use_mega_moe` falls back off the MegaMoE path for large batches.

## The fix

Use the already-resolved per-rank `chunked_prefill_size` in the attention-DP
branch. I also updated the diagnostic label so it states that the DP division
was already applied.

## Verification

I added `test_megamoe_token_budget_uses_resolved_dp_chunked_prefill`. The test
declares an 8192-token per-rank chunk, the state after `handle_data_parallelism`.
It checks that 8192 passes and that 4096 is rejected with
`required_per_rank=8192`.

| Test state | Result |
|:---|:---|
| New test on main (before fix) | fails: `AssertionError: ValueError not raised` |
| New test with fix | passes |
| `pytest ... -k megamoe` | 19 passed |
| `pre-commit run --files ...` | passes |

I also called the real `handle_data_parallelism` hook. It logs
`adjusted from 16384 to 8192` and records `{'chunked_prefill_size': 8192}`.

This change is startup argument validation only. It does not change a model
forward pass or a kernel.
