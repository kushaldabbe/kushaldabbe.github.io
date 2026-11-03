---
title: "Third vLLM PR merged: embedding LoRA ignored embed_scale"
date: 2026-11-03 20:00:00 +0900
---

My third contribution to [vLLM](https://github.com/vllm-project/vllm) merged on
3 November 2026: [PR #62470](https://github.com/vllm-project/vllm/pull/62470),
a LoRA fix for scaled input embeddings.

[![vLLM PR #62470 - merged card](https://opengraph.githubassets.com/1/vllm-project/vllm/pull/62470)](https://github.com/vllm-project/vllm/pull/62470)
{: .kd-og-img }

## The bug

On Transformers-backend models with a scaled input embedding, an `embed_tokens`
LoRA adapter had only `1 / embed_scale` of its effect. This hit VaultGemma, and
Gemma 1 to 4 with `--model-impl transformers`.

The Transformers backend rebases a HuggingFace `*ScaledWordEmbedding` onto
`VocabParallelEmbedding` and keeps its `forward`, which returns
`lookup * embed_scale` (`replace_embedding_class` in
`transformers/utils.py`). `VocabParallelEmbeddingWithLoRA.forward` calls that
`base_layer.forward` and then adds the LoRA delta without the scale.

PEFT's LoRA `Embedding.forward` multiplies the adapter output by `embed_scale`
([huggingface/peft#2825](https://github.com/huggingface/peft/pull/2825),
released in 0.18.0). A merged adapter is also scaled by construction. So vLLM
and PEFT disagreed.

## The fix

In `set_lora`, multiply the slot's `lora_b` by `embed_scale` after the copy to
the device. The scale is applied once, so there is no per-step cost.

Native models with an embedding multiplier (Granite, MiniCPM, Falcon-H1,
HyperCLOVA X) apply the multiplier after the LoRA-wrapped lookup. They do not
change.

## Verification

I used a tiny VaultGemma (random weights saved with `save_pretrained`) and a
PEFT 0.21.1 adapter with `target_modules=["embed_tokens"]`. Both sides used the
same token ids. The PEFT reference was
`PeftModel.get_input_embeddings()(ids)`.

| Metric | main | This PR |
|:---|:---:|:---:|
| LoRA delta, vLLM / PEFT | 0.0625 (`1 / embed_scale`) | 1.0000 |
| Max abs diff vs PEFT embeddings | 1.11 | 2.4e-07 |
| Max abs log-prob diff | 1.18 | 0.000 |

`pytest tests/lora/test_layers.py -k test_embeddings` gives 48 passed. The new
`embed_scale=16.0` cases (24) fail on main with every element mismatched.

The full `tests/lora/test_layers.py` shows the same 102 failures on main and
with this change (`'OffsetReplicatedLinear' object has no attribute
'cpu_linear'`, a local build issue). So the change adds no new failures.

One limitation: adapters trained with PEFT 0.17 or earlier saw an unscaled delta
in the unmerged training forward. This change follows current PEFT and
merged-adapter behavior.
