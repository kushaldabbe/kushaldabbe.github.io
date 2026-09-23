---
title: "Second vLLM PR merged: NaN in the Triton softcap"
date: 2026-09-22 20:00:00 +0900
---

My second contribution to [vLLM](https://github.com/vllm-project/vllm) merged on
22 September 2026: [PR #56579](https://github.com/vllm-project/vllm/pull/56579),
fixing [issue #56578](https://github.com/vllm-project/vllm/issues/56578).

[![vLLM PR #56579 - merged card](https://opengraph.githubassets.com/1/vllm-project/vllm/pull/56579)](https://github.com/vllm-project/vllm/pull/56579)
{: .kd-og-img }

## The bug

`apply_softcap` computed `x * tanh(S / x)` the naive way:

```
x * (exp(S/x) - exp(-S/x)) / (exp(S/x) + exp(-S/x))
```

Both exponentials overflow to `inf` when `|S / x|` exceeds ~88, so the ratio
becomes `inf / inf = NaN` and the whole attention row is poisoned.

This is reachable in production: Gemma-2 style models set
`attn_logit_softcapping = 50` precisely because their attention logits are
very large. A pre-softcap score above ~4400 overflows and the model emits
`NaN` for that row.

## The fix

I rewrote tanh in a form that saturates cleanly instead of exploding:

```
1 - 2 / (exp(2 * S / x) + 1)
```

It stays finite for any input and clips to `+/-1` for either sign. The
helper is shared by the unified, DiffKV, and INT4 attention kernels, so one
fix covers three backends.

## Verification

I verified it on a GTX 1650 Ti with the real helper:

```text
input  : [1.0, 100.0, 1000.0, 3000.0, 5000.0, 10000.0, -1.0, -10000.0, 4400.0]
before : [0.9999, 48.2, 50.0, 50.0, NaN, NaN, -0.9999, NaN, inf]
after  : [0.9999, 48.2, 50.0, 50.0, 50.0, 50.0, -0.9999, -50.0, 50.0]
```

| Test state | Result |
|:---|:---|
| New overflow test on main (before fix) | fails with `NaN` |
| New overflow test with fix | passes |
| 336 existing softcap tests | pass |

The `fp8` cases fail only on SM75, where Triton itself does not support
`fp8e4nv`; a pre-existing environment limitation, unrelated to this change.
