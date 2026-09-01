# EXL3 candidate rejected for batched-choice late JIT

## Identity and disposition

- Image ID: `sha256:b60aeeac44ca95e25192ed4f86a30418ab21332bee184f777ab3e73985c94807`
- Image recipe: `f6d22ed97177cd5e6a446c8c864348180b81cbb5`
- Topology: one DGX Spark (`kiwi`), mixed K2/K3 EXL3
- Profile: fixed greedy native DSpark K5
- Result: **rejected**, not release-qualified

The exact image passed the CUDA-hidden B12x, Vision, and EXL3 contracts. Its
native Vision role also passed cold and warm-cache startup, ordered 1/4/16
image reads, image-17 rejection, two exact 128K six-needle runs, 30 complete
1/4/16-image reliability cycles, and every post-ready JIT audit.

The same image's EXL3 role passed startup, both exact 128K prefix-replay
answers, a 127,744-token real prefix-cache hit, and all 20 post-long-context C4
soak requests. The soak median was 158.19 aggregate tok/s. It is nevertheless
rejected because the final audit observed one compile after readiness:

- Kernel: `_compute_global_topk_indices_and_lens_kernel`
- Trigger: the first release-style `/v1/completions` request with `n=4`
- Post-ready request count at audit: 38
- Service state at audit: healthy

Startup had covered four simultaneous `n=1` requests, but vLLM's OpenAI `n=4`
choice expansion builds a distinct request-to-token mapping and selected a new
DeepSeek sparse-cache Triton specialization. The next candidate adds explicit
temperature-matched `n=2` and `n=4` completions before the ready marker while
retaining the existing independent-request concurrency passes. Its proof must
use a fresh Triton cache so an artifact from this rejected run cannot mask the
specialization.

The machine-readable failed audit is retained in
[`2026-09-02-exl3-batched-n-late-jit.json`](2026-09-02-exl3-batched-n-late-jit.json).
None of the measurements above are release results.
