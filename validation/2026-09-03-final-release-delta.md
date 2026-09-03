# Final v0.1.0 image delta qualification

The release image is
`sha256:dcafc6bf649d70a014ff4350eba85cd7e721dec0ecb9a24ea38bd58401ffe8bd`,
built at `93df1414cd5aa558d7064706e8d37c93651c59c6`. It was independently verified
on `emu` and `kiwi`; all GPU execution remained on SM121 DGX Sparks.

The immediately preceding native image completed the full release suite. The
final source delta adds the missing GB10 mHC startup warmup classes and
backports vLLM's termination-safe xgrammar batch accounting, neither of which
changes the model or ordinary attention/MoE/decode fast path. Repeating the
full matrix would therefore add variance rather than isolate these changes.

The exact final image passed the prior TC31 late-JIT trigger, a short
C1/C2/C4 performance envelope, 12 repeated constrained-JSON requests, five
deterministic tool calls, 1/4/16-image ordering, image-17 rejection, and exact
Vision prefix replay with changed-image isolation. After 40 post-ready
requests, Docker remained healthy and the audit observed zero TileLang/Triton
JIT events, xgrammar/FSM warnings, tracebacks, or server errors.

The exact image was then launched independently on `ostrich` with FP8 and
`dodo` with NVFP4. Both completed strict startup, retained zero post-ready JIT,
and matched the earlier performance suites within 2% at C1 and C4. FP8 exposed
1,762,308 physical KV tokens and NVFP4 exposed 2,039,387, a 15.72% increase.

Raw receipts are retained in
[`benchmarks/20260903T013200Z-final-delta-93df141`](../benchmarks/20260903T013200Z-final-delta-93df141/).
