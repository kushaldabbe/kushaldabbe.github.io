---
title: "Five kernels against SGLang: my Muse Spark Open Kernel Challenge"
date: 2026-11-18 12:00:00 +0900

---

For two weeks (22 October to 4 November 2026) I took part in the kernel track of the
[Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). The task was
to write GPU kernels for one decoder layer of Muse Spark Open and beat SGLang's own
kernels on an H200.

## TL;DR

- **The whole decoder layer ran 1.75× faster than SGLang** (22.62 ms → 12.91 ms per step).
- **MoE was my best board at 2.03×** (13.45 ms → 6.63 ms).
- All five boards finished between 1.26× and 2.03×, for **rank 691 of 2,014** overall.
- That took **665 judged runs**, all remote on their H200.
- The biggest single lesson: one night auditing the hot paths I already had beat a
  week of new kernel designs.

## How the judge worked

Each problem was one piece of the model, as a single GPU of an 8×H200 server runs it.

- **Submission.** One Python file per problem with a `kernel_fn(data, state)` entry
  point. Inline CUDA was allowed, and an untimed `prepare()` step could pack weights or
  tune kernels ahead of time. Anything SGLang ships (Triton, CUTLASS, FlashInfer,
  DeepGEMM) was fair game.
- **Correctness.** Every run used fresh random inputs, checked against an FP32
  reference within a tolerance. Lower precision and lossy weights were banned, so every
  speedup had to come from doing the same math faster.
- **Timing.** My kernel and SGLang's took turns on the same GPU with the same inputs.
  Each turn ran as a CUDA graph, the way SGLang runs in production, and the L2 cache was
  flushed between turns so neither side benefited from data the other left behind.
- **Score.** Each problem was timed under several *loads*: plain decode with 1, 16 and
  32 users, and speculative verify, where each user checks an 8-token draft in one step
  (8, 128 and 256 tokens). The score is the geometric mean of the speedups across those
  loads, so each load counts equally.

## Results

| Problem | SGLang | Mine | Speedup |
|---|---:|---:|---:|
| Whole layer (`muse_decode_layer`) | 22.62 ms | 12.91 ms | **1.75×** |
| MoE (`muse_moe_w4a16`) | 13.45 ms | 6.63 ms | **2.03×** |
| Attention (`muse_attention_decode`) | 3.77 ms | 2.06 ms | **1.83×** |
| Dense projections (`muse_bf16_proj`) | 4.83 ms | 3.85 ms | **1.26×** |
| Prefill layer (`muse_prefill_layer`) | 387.4 ms | 300.2 ms | **1.29×** |

Times are the judge's time per step, averaged over all loads. Being honest about where that lands:
it's the middle of the pack. I beat the field median on MoE and dense projections and
fell below it on the other three. The leaders show how much was left.

<div class="mk-card">
  <div class="mk-cap">Speedup over SGLang, against the field</div>
  <div id="mk-field"></div>
  <div class="mk-legend">
    <span><i class="mk-k-me"></i>mine</span>
    <span><i class="mk-k-med"></i>field median</span>
    <span><i class="mk-k-lead"></i>board leader</span>
  </div>
</div>

## The five problems

Pick a problem to see which part of the layer it covers and how the speedup splits
across loads.

<div class="mk-card">
  <div class="mk-tabs" id="mk-tabs" role="tablist"></div>
  <svg class="mk-svg" id="mk-schem" viewBox="0 0 720 250" role="img" aria-label="Decoder layer diagram"></svg>
  <div class="mk-row" id="mk-ms"></div>
  <div class="mk-cap" id="mk-lcap"></div>
  <svg class="mk-svg" id="mk-loads" viewBox="0 0 720 240" role="img" aria-label="Speedup by load"></svg>
  <div class="mk-legend" id="mk-lleg"></div>
</div>

### Dense projections (`muse_bf16_proj`, 1.26×)

The layer's five weight matmuls (`x @ W.T`): the qkv_gate, o, pooler and unpooler
projections in BF16, plus the router in FP32, at 1 to 256 tokens per step. With few
tokens there is almost no math, so the GPU spends its time streaming weights from
memory. The only way past cuBLAS is to read fewer bytes.

- **12-bit lossless weights.** Most BF16 weights fall in a narrow range of exponents,
  so each one fits in its low byte plus a 4-bit sign-and-exponent code. The ~0.01%
  that don't fit go in a separate correction list. That's 25% fewer bytes with
  bit-identical output.
- **gv5**, a GEMV kernel that unpacks the 12-bit stream straight into tensor-core
  (`mma.sync`) inputs. With no H200 to test on, I first checked the unpacking
  bit-for-bit with a CPU emulation.
- **The router as two BF16 halves** (hi + lo), so the FP32 router can use the fast
  BF16 tensor cores and still pass the FP32 accuracy check: 1.4–2.6× on every load.
- **A wgmma kernel for large batches.** From m ≥ 128 the work becomes compute-bound
  and cuBLAS is hard to beat. Mine only matched it, and that's where the leaders found
  their extra 0.4×.

### Attention decode (`muse_attention_decode`, 1.83×)

Each new token attends over an FP8 KV cache of about 80K tokens. Sliding-window
layers read only the most recent part of that cache; full layers read all of it. Both
are timed in plain decode and in 8-token speculative verify.

- **swa**, a fused sliding-window kernel. A cluster of thread blocks splits the window
  and merges their partial softmax results in shared memory instead of going through
  HBM. Batch-1 decode went from 25.0 to 10.5 µs.
- **attn12** for full layers. It splits the long context across many CTAs with FP8
  wgmma, and a small combine kernel merges the partial results.
- **A hot-path audit.** I swapped the library `exp2f` for the hardware `ex2.approx`,
  sized ring buffers as powers of two so every `%` became a bit mask, and moved setup
  out of the inner loop. That alone took the board from 1.57× to 1.71×.
- **A per-shape tuner** in `prepare()` that tests my kernels against FA3 (SGLang's
  kernel) and keeps the fastest one that passes an FP32 check.

### MoE (`muse_moe_w4a16`, 2.03×)

256 experts with NVFP4 (4-bit) weights, split across 8 GPUs so each holds 32. Every
token goes to its top 8 experts, so each step means: route tokens to experts, run two
GEMMs per expert, then combine the results.

- **moe4** runs that pipeline (route → GEMM1 → GEMM2 → finalize), decoding FP4 to BF16
  exactly as the weights stream in.
- **Independent warps.** Each warp keeps its own `cp.async` load queue and never waits
  on the rest of its block. From batch 32 up, the kernel runs as fast as the weight
  loads alone, which is the memory limit.
- **Early GEMM2 loads.** GEMM2's weights don't depend on GEMM1's output, so they start
  loading before GEMM1 finishes, saving 2–7 µs per call.
- **A per-batch config tuner,** plus a faster routing kernel (6.0 → 4.3 µs).

### Whole decoder layer (`muse_decode_layer`, 1.75×)

The full layer: the projections, attention and MoE above, plus the RMSNorms and
residual adds between them. My kernels from the other problems are its parts; the rest
of the win comes from shrinking the gaps between kernels.

- **Reusing the kernels above** (gv5, swa/attn12, moe4), each picked per shape.
- **Fused boundaries.** I ported SGLang's faster norm-residual kernel (4.5 → 2.5 µs)
  and folded one RMSNorm into GEMM1, so it runs while the input loads.
- **Overlap.** Expert routing runs on a side stream next to the pooler, and PDL
  (programmatic dependent launch) starts each kernel before the previous one
  finishes, hiding its setup.
- **A race I introduced.** One early-launched kernel read its input before the
  previous kernel had finished writing it. Tests still passed because the buffer held
  identical values from the last call. I fixed it and confirmed the score hadn't
  depended on it.

### Prefill layer (`muse_prefill_layer`, 1.29×)

The same layer on whole prompts: 753-token turns and 16K-token first turns. With this
many tokens the MoE GEMMs dominate. I set it aside after day one and only came back on
day 8.

- **moe15**, a grouped wgmma GEMM that decodes each expert's weights once per call and
  reuses them for all of its tokens. It replaced Marlin, which had been about 65% of
  the time.
- **At 16K tokens,** dequantize all experts to BF16 once, then run DeepGEMM's grouped
  GEMM.
- **Custom routing** in place of PyTorch's argsort and scatter: the MoE part at 16K
  went from 5.5 to 5.0 ms.

## Two weeks of progress

The best judged score so far on each board. The shaded band marks the hot-path audit.
Before it came a week of new kernel designs that mostly didn't beat what I had.

<div class="mk-multi" id="mk-multi"></div>

Each point is a judged run: both scored runs, which post to the board, and unscored
checks, which test code without posting. Run-to-run noise is about ±2%, so a curve can
end slightly above its final board score.

## Where the time goes

This is one decoder layer at batch 1 (a single user), on a sliding-window layer with 80K
tokens of context. For every shape, the judge also reports a *speed of light*: the time
it would take just to read every byte the step needs at full H200 memory bandwidth. A
single-user decode step is almost all weight reads, so the distance from that line is
the honest measure of what's left.

<div class="mk-card">
  <svg class="mk-svg" id="mk-time" viewBox="0 0 720 196" role="img" aria-label="Time breakdown at decode batch 1"></svg>
  <div class="mk-legend">
    <span><i class="mk-k-sl"></i>streaming weights (projections, experts)</span>
    <span><i class="mk-k-oth"></i>other kernels</span>
  </div>
</div>

| Kernel group (from the judge's profile) | Time |
|---|---:|
| Dense projections (qkv_gate, o, pooler, unpooler) | 39.0 µs |
| MoE GEMMs (this GPU's experts) | 26.3 µs |
| Expert routing + finalize (sort tokens to experts, combine their outputs) | 8.4 µs |
| Router matmul (scores every expert for each token) | 7.0 µs |
| Attention (swa) | 6.1 µs |
| Norm + residual | 5.1 µs |

The kernels add up to about 92 µs, yet the measured step is 86 µs: with PDL, each kernel
starts its setup (reading parameters, issuing its first memory loads) while the previous
one is still finishing. About three quarters of the time goes to streaming weights. The
rest is a dozen kernels of only a few µs each, where launch overhead and the time to get
memory requests flowing cost as much as the data itself.

## Lessons

1. **Audit before you redesign.** Small inefficiencies in the inner loop cost more than
   I expected: a library `exp2f` where the hardware `ex2.approx` would do, a `%` or `/`
   on every ring-buffer index, and accumulators indexed at runtime that spilled to local
   memory. Fixing them in one night took attention from 1.57× to 1.71×, more than a week
   of new kernel designs had.
2. **The noise is ±2%.** The same code scored anywhere from 1.247 to 1.268 on different
   runs, while most late ideas were worth about 1%. Judge a change by per-shape
   microseconds over several runs, never by one score.
3. **Bandwidth needs bytes in flight.** Memory latency is high, so full bandwidth means
   keeping many reads outstanding at once. One 8-warp thread block per SM streamed only
   2.3–2.6 TB/s. Giving each warp its own `cp.async` or TMA load queue, with no
   block-wide barriers holding it back, reached 4+ TB/s.
4. **PDL is free speed until it's a race.** PDL lets a kernel start before the previous
   one finishes, but it must not read that kernel's output before
   `griddepcontrol.wait`. If it does, reused buffers can still hold the right values
   from the last call, so every test passes anyway.
5. **Clusters can't span GPCs.** The H200's 132 SMs are grouped into 8 GPCs of 16–17,
   and a thread-block cluster must fit inside one GPC. A grid of 33 four-block clusters
   left the last one waiting for a second wave, which doubled the kernel time.
6. **One crash loses the whole run.** An illegal memory access in any candidate the
   autotuner tries corrupts the CUDA context, and then even the fallback kernels fail.
   A tuner protects you from slow candidates, not broken ones.

## Ideas that didn't make it

| Idea | What happened |
|---|---|
| TMA multicast of `x` | Load the input once per cluster and broadcast it to every block, to cut memory traffic. It worked, but each multicast cost ~0.3 µs: +50% time at cluster size 2, +150–200% at 4–8. |
| 12-bit weights for wgmma | Unpack 12-bit weights into shared memory so the faster wgmma instructions could use them at large batches. The unpacked tiles left too little shared memory for the load pipeline: 27–31 µs vs 20–22 µs for plain BF16. |
| Self-routing MoE | Build the expert routing table inside every GEMM block, to skip one kernel launch. Rebuilding it in every block cost more than the launch saved: 73.4 vs 70.8 µs. |
| Fused attention merge | Merge the partial results from the split context inside the same kernel, using a grid-wide barrier, to drop the separate combine kernel. Slower (21.9 vs 20.2 µs), and the barrier hung when reused across launches with different grid sizes. |
| Early PDL triggers everywhere | Let every kernel signal the next one to start as early as possible. It caused occasional NaNs in full-context verify, and without a clear cause I didn't ship it. |
| MegaMoE | DeepGEMM's fused all-in-one MoE kernels. The judge image's builds only supported Blackwell (SM100) or FP8, and lower precision was banned. |

<style>
.mk-card{border:1px solid var(--btn-border-color,#e9ecef);border-radius:.75rem;padding:1rem 1.1rem 1.1rem;margin:1.2rem 0 1.6rem;background:var(--card-bg,transparent)}
.mk-cap{font-size:.8rem;font-weight:600;letter-spacing:.03em;text-transform:uppercase;color:var(--text-muted-color);margin:.2rem 0 .6rem}
.mk-legend{display:flex;flex-wrap:wrap;gap:.35rem 1.2rem;font-size:.82rem;color:var(--text-muted-color);margin-top:.5rem}
.mk-legend i{display:inline-block;vertical-align:-1px;margin-right:.4rem}
.mk-k-me{width:11px;height:11px;border-radius:50%;background:var(--link-color)}
.mk-k-med{width:9px;height:9px;border-radius:50%;background:var(--text-muted-color)}
.mk-k-lead{width:11px;height:11px;border-radius:50%;border:2px solid var(--text-color)}
.mk-k-oth{width:11px;height:11px;border-radius:2px;background:var(--text-muted-color);opacity:.55}
.mk-k-ref{width:16px;height:0;border-top:2px dashed var(--text-muted-color)}
.mk-k-sl{width:11px;height:11px;border-radius:2px;background:var(--link-color)}
.mk-k-fu{width:11px;height:11px;border-radius:2px;background:var(--link-color);opacity:.45}
.mk-svg{width:100%;height:auto;display:block}
.mk-svg text{font-family:inherit;fill:var(--text-color)}
.mk-svg .mu{fill:var(--text-muted-color)}
.mk-svg .ac{fill:var(--link-color)}
.mk-svg .grid{stroke:var(--text-muted-color);stroke-opacity:.25}
.mk-svg .ref{stroke:var(--text-muted-color);stroke-dasharray:4 4}
.mk-svg .wire{stroke:var(--text-muted-color);fill:none}
.mk-svg .wire.on{stroke:var(--link-color)}
.mk-svg .blk{fill:var(--main-bg);stroke:var(--text-muted-color);stroke-width:1}
.mk-svg .blk.on{fill:color-mix(in srgb,var(--link-color) 14%,var(--main-bg));stroke:var(--link-color);stroke-width:1.6}
.mk-svg text.on{fill:var(--link-color)}
.mk-svg .bar{fill:var(--link-color)}
.mk-svg .bar.lite{fill-opacity:.45}
.mk-svg .oth{fill:var(--text-muted-color);fill-opacity:.55}
.mk-svg .sg{fill:var(--text-muted-color);fill-opacity:.3}
.mk-svg .sol{fill:none;stroke:var(--text-muted-color);stroke-dasharray:4 3}
.mk-svg .halo{stroke:var(--main-bg);stroke-width:4px;paint-order:stroke;stroke-linejoin:round}
.mk-svg .band{fill:var(--link-color);fill-opacity:.1}
.mk-svg .line{fill:none;stroke:var(--link-color);stroke-width:2}
.mk-tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.8rem}
.mk-tabs button{border:1px solid var(--btn-border-color,#e0e0e0);background:transparent;color:var(--text-color);border-radius:2rem;padding:.25rem .8rem;font-size:.85rem;cursor:pointer}
.mk-tabs button:hover{border-color:var(--link-color)}
.mk-tabs button[aria-selected="true"]{background:var(--link-color);border-color:var(--link-color);color:var(--main-bg)}
.mk-row{display:flex;flex-wrap:wrap;gap:.3rem 1.6rem;font-size:.9rem;margin:.6rem 0 1rem;padding:.6rem 0;border-top:1px solid var(--btn-border-color,#eee);border-bottom:1px solid var(--btn-border-color,#eee)}
.mk-row b{font-weight:600}
.mk-row .ac{color:var(--link-color);font-weight:600}
.mk-multi{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;margin:1rem 0}
.mk-multi figure{margin:0;border:1px solid var(--btn-border-color,#eee);border-radius:.6rem;padding:.55rem .6rem .3rem}
.mk-multi figcaption{display:flex;justify-content:space-between;font-size:.8rem;color:var(--text-muted-color);margin-bottom:.2rem}
.mk-multi figcaption b{color:var(--link-color);font-weight:600}
@media (max-width:576px){.mk-multi{grid-template-columns:repeat(2,minmax(0,1fr))}.mk-card{padding:.8rem .7rem}}
</style>

{% raw %}
<script>
(function(){
const DATA={"muse_bf16_proj":[["decode 1","qkv_gate",18.3,14.4],["decode 1","o",11.1,10.9],["decode 1","pooler",25.7,19.6],["decode 1","unpooler",11.1,9.3],["decode 1","router",14.5,11.3],["decode 16","qkv_gate",18.3,16.4],["decode 16","o",11.0,10.7],["decode 16","pooler",25.6,20.8],["decode 16","unpooler",22.8,21.9],["decode 16","router",19.1,11.9],["decode 32","qkv_gate",18.8,18.7],["decode 32","o",11.1,10.8],["decode 32","pooler",26.3,23.6],["decode 32","unpooler",23.0,22.1],["decode 32","router",19.5,13.6],["verify 1","qkv_gate",18.5,14.7],["verify 1","o",11.0,10.7],["verify 1","pooler",25.7,19.3],["verify 1","unpooler",11.3,9.5],["verify 1","router",30.5,11.6],["verify 16","qkv_gate",21.6,20.7],["verify 16","o",11.6,11.6],["verify 16","pooler",29.1,27.0],["verify 16","unpooler",24.0,23.1],["verify 16","router",20.6,14.4],["verify 32","qkv_gate",23.7,24.0],["verify 32","o",13.4,13.6],["verify 32","pooler",34.6,32.5],["verify 32","unpooler",28.5,28.6],["verify 32","router",21.8,15.3]],
"muse_attention_decode":[["decode 1","sliding",25.0,10.5],["decode 1","full",35.6,19.4],["decode 16","sliding",30.7,14.3],["decode 16","full",205.0,109.1],["decode 32","sliding",35.4,22.5],["decode 32","full",388.8,194.5],["verify 1","sliding",27.0,13.9],["verify 1","full",40.4,36.5],["verify 16","sliding",33.3,24.0],["verify 16","full",209.0,151.3],["verify 32","sliding",38.6,33.1],["verify 32","full",393.0,269.2]],
"muse_moe_w4a16":[["decode 1","bs 1",81.5,33.3],["decode 16","bs 16",189.6,99.7],["decode 32","bs 32",208.3,133.6],["verify 1","bs 8",147.0,74.6],["verify 16","bs 128",461.9,215.1],["verify 32","bs 256",635.0,284.6]],
"muse_decode_layer":[["decode 1","sliding",175.6,86.0],["decode 1","full",187.5,101.5],["decode 16","sliding",263.2,157.8],["decode 16","full",434.0,255.3],["decode 32","sliding",266.7,215.1],["decode 32","full",628.8,383.1],["verify 1","sliding",262.3,124.8],["verify 1","full",275.6,149.7],["verify 16","sliding",520.8,310.0],["verify 16","full",704.8,437.0],["verify 32","sliding",732.9,427.0],["verify 32","full",1084.1,658.7]],
"muse_prefill_layer":[["turn 1","sliding",1094.5,883.1],["turn 1","full",1509.6,1305.0],["turn 4","sliding",3358.5,2685.0],["turn 4","full",5019.4,4328.8],["16K · ctx 0","full",16901.5,11557.2],["16K · ctx 32K","sliding",17168.9,11405.3],["16K · ctx 32K","full",21588.2,15717.2],["16K · ctx 64K","full",24751.2,19057.8]]};
const CURVE={"muse_bf16_proj":[[0,1],[0,1.051],[0.48,1.097],[0.81,1.126],[2.75,1.146],[7.53,1.176],[7.8,1.228],[7.81,1.242],[8.93,1.245],[9.94,1.252],[10.79,1.254],[11.89,1.268]],
"muse_attention_decode":[[0,1],[0.33,1.177],[0.43,1.242],[0.64,1.311],[1.48,1.344],[1.54,1.423],[1.55,1.434],[2.81,1.526],[4.81,1.558],[7.34,1.573],[7.49,1.649],[7.5,1.683],[7.61,1.721],[8.32,1.781],[8.34,1.799],[8.37,1.823],[11.78,1.827]],
"muse_moe_w4a16":[[0,1],[0.26,1.165],[0.77,1.314],[0.8,1.437],[0.82,1.512],[0.93,1.544],[1.29,1.578],[2.71,1.608],[7.34,1.71],[7.55,1.737],[7.79,1.807],[7.8,1.823],[8.68,1.859],[8.84,1.887],[9.75,1.922],[9.95,1.96],[10.78,2.011],[11.82,2.029]],
"muse_decode_layer":[[0,1],[0.33,1.111],[0.73,1.159],[0.84,1.28],[1.54,1.325],[2.72,1.384],[4.73,1.403],[7.34,1.452],[7.53,1.489],[7.58,1.495],[7.8,1.595],[8.81,1.643],[9.87,1.673],[9.95,1.695],[10.78,1.725],[11.19,1.736],[11.88,1.751]],
"muse_prefill_layer":[[0,1],[0.79,1.01],[7.64,1.115],[7.69,1.141],[7.9,1.161],[7.91,1.239],[8.35,1.256],[8.88,1.279],[8.91,1.29]]};
const P=[
 {id:"muse_decode_layer",short:"Decode layer",score:1.7523,med:1.786,lead:2.6144,rank:688,of:1299,base:"22.62 ms",ours:"12.91 ms",hl:"*",mode:"sf"},
 {id:"muse_moe_w4a16",short:"MoE",score:2.0291,med:1.9166,lead:2.8438,rank:635,of:1425,base:"13.45 ms",ours:"6.63 ms",hl:["experts"],mode:"moe"},
 {id:"muse_attention_decode",short:"Attention",score:1.8266,med:1.9069,lead:3.3389,rank:745,of:1379,base:"3.77 ms",ours:"2.06 ms",hl:["attn"],mode:"sf"},
 {id:"muse_bf16_proj",short:"Dense projections",score:1.2561,med:1.2467,lead:1.7277,rank:843,of:1746,base:"4.83 ms",ours:"3.85 ms",hl:["qkv","o","router","pooler","unpooler"],mode:"proj"},
 {id:"muse_prefill_layer",short:"Prefill layer",score:1.2906,med:1.6041,lead:2.1471,rank:674,of:898,base:"387.4 ms",ours:"300.2 ms",hl:"*",mode:"sf"}];
const NS="http://www.w3.org/2000/svg";
function el(t,a,p,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;if(p)p.appendChild(e);return e}
function tip(e,t){el("title",{},e,t);return e}
const gm=a=>Math.exp(a.reduce((s,x)=>s+Math.log(x),0)/a.length);
function rbar(x,y,w,h,r){r=Math.min(r,w/2,h);return `M${x} ${y+h}V${y+r}Q${x} ${y} ${x+r} ${y}H${x+w-r}Q${x+w} ${y} ${x+w} ${y+r}V${y+h}Z`}
/* field dot plot */
function field(){
  const host=document.getElementById("mk-field");const W=720,rh=40,H=P.length*rh+46,L=150,R=690;
  const s=el("svg",{class:"mk-svg",viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Speedup per board versus median and leader"},host);
  const X=v=>L+(v-1)/(3.5-1)*(R-L);
  [1,1.5,2,2.5,3,3.5].forEach(v=>{el("line",{x1:X(v),y1:6,x2:X(v),y2:H-28,class:v===1?"ref":"grid"},s);el("text",{x:X(v),y:H-10,"font-size":12,"text-anchor":"middle",class:"mu"},s,v+"×")});
  P.forEach((p,i)=>{const y=30+i*rh;
    el("text",{x:0,y:y+4,"font-size":13.5},s,p.short);
    el("line",{x1:X(1),y1:y,x2:X(p.lead),y2:y,class:"grid","stroke-width":2},s);
    tip(el("circle",{cx:X(p.med),cy:y,r:4.5,class:"mu"},s),`field median ${p.med.toFixed(2)}×`);
    tip(el("circle",{cx:X(p.lead),cy:y,r:6,fill:"none",stroke:"currentColor","stroke-width":2,style:"color:var(--text-color)"},s),`leader ${p.lead.toFixed(2)}×`);
    tip(el("circle",{cx:X(p.score),cy:y,r:7,class:"ac",stroke:"var(--main-bg)","stroke-width":2},s),`mine ${p.score.toFixed(2)}× · rank ${p.rank} of ${p.of}`);
    el("text",{x:X(p.score),y:y-11,"font-size":12.5,"text-anchor":"middle",class:"ac halo","font-weight":600},s,p.score.toFixed(2)+"×");
  });
}
/* schematic */
function schem(p){
  const s=document.getElementById("mk-schem");s.innerHTML="";
  const all=p.hl==="*",on=n=>all||p.hl.includes(n);
  const box=(n,x,y,w,h,lab)=>{el("rect",{x,y,width:w,height:h,rx:6,class:"blk"+(on(n)?" on":"")},s);el("text",{x:x+w/2,y:y+h/2+5,"font-size":14,"text-anchor":"middle",class:on(n)?"on":""},s,lab)};
  const wire=(d,n)=>el("path",{d,class:"wire"+(n&&on(n)?" on":"")},s);
  const plus=(x,y)=>{el("circle",{cx:x,cy:y,r:9,class:"blk"+(all?" on":"")},s);el("path",{d:`M${x-5} ${y}h10M${x} ${y-5}v10`,class:"wire"+(all?" on":""),"stroke-width":1.5},s)};
  el("text",{x:0,y:16,"font-size":12,class:"mu"},s,"attention block");
  el("text",{x:0,y:136,"font-size":12,class:"mu"},s,"MoE block");
  const y1=62,y2=192;
  wire(`M0 ${y1}H40`);box("qkv",40,y1-20,100,40,"qkv_gate");wire(`M140 ${y1}H180`);
  box("attn",180,y1-26,150,52,"attention");
  el("text",{x:255,y:y1+44,"font-size":11.5,"text-anchor":"middle",class:on("attn")?"on":"mu"},s,"FP8 KV cache · 80K tokens");
  wire(`M330 ${y1}H370`,"attn");box("o",370,y1-20,80,40,"o");
  wire(`M450 ${y1}H551`);el("text",{x:500,y:y1-8,"font-size":11.5,"text-anchor":"middle",class:all?"on":"mu"},s,"norms");
  plus(560,y1);wire(`M569 ${y1}H710V112H10V${y2}H30`);
  wire(`M30 ${y2}V${y2-20}H44M30 ${y2}V${y2+20}H44`);
  box("router",44,y2-36,86,30,"router");box("pooler",44,y2+6,86,30,"pooler");
  wire(`M130 ${y2+21}H150`);
  const ex=on("experts"),act=[3,7,12,16,21,24,27,30];
  for(let i=0;i<32;i++){const a=act.includes(i);el("rect",{x:152+i*7.2,y:y2-22,width:5.2,height:44,rx:1.5,class:"blk"+(a&&ex?" on":""),"stroke-width":a&&ex?1.4:.8},s)}
  el("text",{x:267,y:y2+40,"font-size":11.5,"text-anchor":"middle",class:ex?"on":"mu"},s,"256 NVFP4 experts · top-8 · 32 here");
  wire(`M384 ${y2}H420`,"experts");box("unpooler",420,y2-20,96,40,"unpooler");
  wire(`M516 ${y2}H551`);plus(560,y2);wire(`M569 ${y2}H720`);
  const m=document.getElementById("mk-ms");
  m.innerHTML=`<span>SGLang <b>${p.base}</b></span><span>mine <b>${p.ours}</b></span><span>speedup <span class="ac">${p.score.toFixed(2)}×</span></span><span>rank <b>${p.rank}</b> of ${p.of.toLocaleString()}</span><span>leader <b>${p.lead.toFixed(2)}×</b></span>`;
}
/* per-load bars */
function loads(p){
  const s=document.getElementById("mk-loads");s.innerHTML="";const rows=DATA[p.id];let groups;
  const cap=document.getElementById("mk-lcap"),leg=document.getElementById("mk-lleg");
  if(p.mode==="proj"){cap.textContent="Speedup by projection (geometric mean over the six loads)";
    groups=[{label:"",bars:["qkv_gate","o","pooler","unpooler","router"].map(n=>{const r=rows.filter(q=>q[1]===n);return{lab:n,v:gm(r.map(q=>q[2]/q[3])),t:`${n}: ${gm(r.map(q=>q[2]/q[3])).toFixed(2)}× over six loads`}})}];
    leg.innerHTML=`<span><i class="mk-k-ref"></i>SGLang = 1.0×</span>`;
  }else if(p.mode==="moe"){cap.textContent="Speedup by load (tokens per step below each bar)";
    groups=[{label:"decode",bars:rows.slice(0,3)},{label:"8-token verify",bars:rows.slice(3)}].map(g=>({label:g.label,bars:g.bars.map(r=>({lab:r[1],v:r[2]/r[3],t:`${r[0]}, ${r[1]}: ${r[2]} → ${r[3]} µs`}))}));
    leg.innerHTML=`<span><i class="mk-k-ref"></i>SGLang = 1.0×</span>`;
  }else{cap.textContent="Speedup by load";
    const ls=[...new Set(rows.map(r=>r[0]))];
    groups=ls.map(l=>({label:l,bars:rows.filter(r=>r[0]===l).map(r=>({lite:r[1]==="full",v:r[2]/r[3],t:`${l}, ${r[1]}: ${r[2]} → ${r[3]} µs`}))}));
    leg.innerHTML=`<span><i class="mk-k-sl"></i>sliding-window layer</span><span><i class="mk-k-fu"></i>full-attention layer</span><span><i class="mk-k-ref"></i>SGLang = 1.0×</span>`;}
  const W=720,top=24,base=190;const vmax=Math.max(2.6,Math.max(...groups.flatMap(g=>g.bars.map(b=>b.v)))*1.15);
  const Y=v=>base-v/vmax*(base-top);
  const nb=groups.reduce((a,g)=>a+g.bars.length,0),gap=18,inner=2;
  const bw=Math.min(70,(W-gap*(groups.length-1)-inner*(nb-groups.length))/nb);
  const tot=nb*bw+inner*(nb-groups.length)+gap*(groups.length-1);let x=(W-tot)/2;
  for(let v=0.5;v<vmax;v+=0.5)el("line",{x1:0,y1:Y(v),x2:W,y2:Y(v),class:v===1?"ref":"grid"},s);
  const labs=[];
  groups.forEach(g=>{const gx=x;
    g.bars.forEach((b,bi)=>{tip(el("path",{d:rbar(x,Y(b.v),bw,base-Y(b.v),4),class:"bar"+(b.lite?" lite":"")},s),b.t);
      labs.push([x+bw/2,Y(b.v)-6,b.v.toFixed(2)+"×",bw]);
      if(b.lab)el("text",{x:x+bw/2,y:base+16,"font-size":bw<50?11:12.5,"text-anchor":"middle",class:"mu"},s,b.lab);
      x+=bw+(bi<g.bars.length-1?inner:0)});
    if(g.label)el("text",{x:(gx+x)/2,y:base+(g.bars.some(b=>b.lab)?34:18),"font-size":12.5,"text-anchor":"middle"},s,g.label);
    x+=gap});
  labs.forEach(([lx,ly,t,bw])=>el("text",{x:lx,y:ly,"font-size":bw<34?10:12,"text-anchor":"middle",class:"halo"},s,t));
  el("line",{x1:0,y1:base,x2:W,y2:base,class:"wire"},s);
}
function tabs(){
  const host=document.getElementById("mk-tabs");
  P.forEach((p,i)=>{const b=document.createElement("button");b.type="button";b.setAttribute("role","tab");b.textContent=p.short;
    b.onclick=()=>pick(i);host.appendChild(b)});
  pick(0);
}
function pick(i){document.querySelectorAll("#mk-tabs button").forEach((b,j)=>b.setAttribute("aria-selected",j===i));schem(P[i]);loads(P[i])}
/* progress small multiples */
function multi(){
  const host=document.getElementById("mk-multi");
  P.forEach(p=>{const f=document.createElement("figure");f.innerHTML=`<figcaption><span>${p.short}</span><b>${p.score.toFixed(2)}×</b></figcaption>`;host.appendChild(f);
    const s=el("svg",{class:"mk-svg",viewBox:"0 0 220 130",role:"img","aria-label":p.short+" best score over time"},f);
    const L=28,R=214,T=6,B=108,X=d=>L+d/13.2*(R-L),Y=v=>B-(v-0.9)/(2.2-0.9)*(B-T);
    el("rect",{x:X(7.3),y:T,width:X(8.4)-X(7.3),height:B-T,class:"band"},s);
    [1,1.5,2].forEach(v=>{el("line",{x1:L,y1:Y(v),x2:R,y2:Y(v),class:v===1?"ref":"grid"},s);el("text",{x:L-4,y:Y(v)+3.5,"font-size":10,"text-anchor":"end",class:"mu"},s,v+"×")});
    let d=`M${X(0)} ${Y(1)}`;CURVE[p.id].forEach(([t,v])=>d+=`H${X(t).toFixed(1)}V${Y(v).toFixed(1)}`);d+=`H${X(12.6)}`;
    el("path",{d,class:"line"},s);
    [["Oct 22",0,"start"],["Oct 29",7,"middle"],["Nov 4",13,"end"]].forEach(([t,dd,a])=>el("text",{x:X(dd),y:B+16,"font-size":10,"text-anchor":a,class:"mu"},s,t));
  });
}
/* time breakdown */
function timebar(){
  const s=document.getElementById("mk-time");const L=130,R=712,X=v=>(v/175.6)*(R-L);
  const row=(y,lab,sub)=>{el("text",{x:0,y:y+15,"font-size":13.5},s,lab);el("text",{x:0,y:y+32,"font-size":11.5,class:"mu"},s,sub)};
  row(8,"SGLang","175.6 µs");tip(el("rect",{x:L,y:8,width:X(175.6),height:30,rx:4,class:"sg"},s),"SGLang 175.6 µs");
  const segs=[["Dense projections",39.0,1],["MoE GEMMs",26.3,1],["Routing + finalize",8.4,0],["Router",7.0,0],["Attention",6.1,0],["Norm + residual",5.1,0]];
  const scale=86.0/segs.reduce((a,q)=>a+q[1],0);
  row(66,"Mine","86.0 µs · 2.04×");let x=L;
  segs.forEach(([n,v,w])=>{const wd=X(v*scale)-2;tip(el("rect",{x,y:66,width:wd,height:30,rx:3,class:w?"bar":"oth"},s),`${n}: ${v} µs (profile)`);
    if(wd>60)el("text",{x:x+6,y:86,"font-size":11.5,fill:"var(--main-bg)",style:"fill:var(--main-bg)"},s,n.split(" ")[0]+" "+v);x+=wd+2});
  row(124,"Speed of light","40.0 µs");tip(el("rect",{x:L,y:124,width:X(40),height:30,rx:4,class:"sol"},s),"bytes ÷ H200 bandwidth: 40.0 µs");
  el("text",{x:L+X(40)+8,y:144,"font-size":11.5,class:"mu"},s,"every byte read once at full HBM bandwidth");
}
function init(){field();tabs();multi();timebar()}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init);else init();
})();
</script>
{% endraw %}
