---
title: vLLM vs SGLang — Benchmarking LLM Serving Engines on an RTX 4090
date: 2026-08-16 00:00:00 +0900
pin: true
---

I benchmarked [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) on a single RTX 4090, serving `meta-llama/Meta-Llama-3-8B-Instruct` in fp16. I wanted to see how the two leading open-source serving engines compare under identical load. The harness is in [inference-bench](https://github.com/kushaldabbe/inference-bench), with all raw data checked in.

## TL;DR

- At **concurrency 1**, both engines give roughly 56-60 tok/s. That is the memory-bandwidth limit for an 8B fp16 model on a 4090 (about 1000 GB/s of HBM divided by 16 GB of weights).
- Throughput scales about 18x with concurrency via **continuous batching**, reaching ~1000-1060 tok/s at 32 concurrent requests.
- **vLLM and SGLang were effectively tied** (within ~1-2% at every concurrency level).
- 960 requests per engine, **0% errors**, every response exactly 128 tokens.

## Setup

| | |
|---|---|
| Hardware | RTX 4090 (24 GB), RunPod, driver 580 / CUDA 13.0 |
| Model | `meta-llama/Meta-Llama-3-8B-Instruct`, fp16, greedy (`temperature=0`) |
| Engines | vLLM 0.11.0, SGLang (FlashInfer attention backend) |
| Workload | concurrency `{1,2,4,8,16,32}` x prompt len `{32,128,512,2048}` x 40 requests = 960 requests/engine |
| API | `/v1/chat/completions`, streaming (OpenAI-compatible) |
| Output | `max_tokens=128`, all responses exactly 128 tokens |

Each engine ran in its own virtualenv on the same pod, so the two stacks never fought over torch or dependency versions. Prompts came from a fixed seed, so both engines processed the exact same requests.

## Results

![Throughput vs concurrency]({{ '/assets/img/benchmark_throughput_vs_concurrency.png' | relative_url }})

| Concurrency | vLLM tok/s | SGLang tok/s |
|---|---|---|
| 1 | 58.6 | 60.2 |
| 2 | 115.4 | 118.1 |
| 4 | 230.3 | 234.9 |
| 8 | 454.5 | 464.8 |
| 16 | 738.6 | 762.3 |
| 32 | 1061.4 | 1077.7 |

(Averaged across prompt lengths. Per-cell TTFT/ITL/E2E percentiles are in the [results README](https://github.com/kushaldabbe/inference-bench/tree/main/results).)

Latency stayed flat under load. Time to first token held steady until the highest concurrency, and the end-to-end time for a 128-token response barely moved even as throughput grew ~18x.

![TTFT vs concurrency]({{ '/assets/img/benchmark_ttft_vs_concurrency.png' | relative_url }})

![E2E latency vs concurrency]({{ '/assets/img/benchmark_e2e_vs_concurrency.png' | relative_url }})

### Reading the curves

- **At concurrency 1 you are bandwidth-limited.** An 8B fp16 model is ~16 GB of weights. The 4090 reads it at ~1000 GB/s, which caps decode near ~60 tok/s. Both engines hit that ceiling.
- **Continuous batching does the heavy lifting.** Each engine packs many concurrent decodes onto the GPU, which is how aggregate throughput reaches ~1000 tok/s at concurrency 32. Per-request e2e latency stayed ~2.1-2.5 s for a full response the whole time, so the extra load showed up as throughput, not latency.
- **TTFT holds up well.** It sat around 28-40 ms through concurrency 16 and only rose to 66-90 ms at 32, when the prefill queue starts to form. So an interactive user still gets a fast first token while the GPU is busy.
- **Raw throughput does not separate these two.** On this workload neither engine had a real advantage. The differences that matter are operational: prefix caching, speculative decoding, guided decoding, memory management, and ecosystem.

## Things I got wrong first (and fixed)

Two measurement problems showed up while I was validating the harness. I want to record them because both silently corrupt a benchmark if you do not catch them.

1. **Output length, not max_tokens.** My first prompt said *"Summarize the above in one sentence."* The model did exactly that and stopped early via EOS at ~30-44 tokens, so I was comparing throughput of short answers rather than a fixed-length workload. I switched to a *"write a long, detailed analysis"* instruction so every response runs the full 128 tokens. The bad run is archived, not published.

2. **The throughput formula.** My first version computed throughput as `total_tokens / slowest_single_request`. That over-counts badly under concurrency, since it ignores that all the requests in a cell ran in parallel. The fix records `cell_wall_s` (first request start to last request finish) and computes `total_tokens / cell_wall`. That gives the physically plausible ~60 tok/s at concurrency 1; the old formula claimed a nonsense 1500 tok/s for the same model.

## Running it yourself

```bash
git clone https://github.com/kushaldabbe/inference-bench.git
cd inference-bench

# deploy a RunPod RTX 4090 pod, then on the pod:
huggingface-cli login
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
CONCURRENCY="1 8 32" PROMPT_LENS="128 512" PROMPTS_PER_CELL=5 \
  ./run_all.sh vllm    # smoke test first

./run_all.sh            # full sweep: vllm, sglang
```

The harness puts each engine in its own venv, gates concurrency with an `asyncio.Semaphore` (an actual in-flight limit, not an httpx connection-pool side effect), and writes per-request JSONL plus aggregated summaries.

## Takeaway

For single-GPU serving of a standard 8B chat model, throughput is not the deciding factor between vLLM and SGLang. Both are fast and both saturate the hardware. Pick based on features (prefix caching, speculative decoding, guided decoding, tool calling), on which one your stack already integrates with, or on what your team is comfortable running. That is the honest result of this benchmark, and it is useful because the numbers now back it up.
