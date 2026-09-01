# Release harness smoke (not release results)

This validates the benchmark clients before freezing the production image. It
does not qualify the current development image and none of these tiny samples
may be reported as release performance.

## Safety and source checks

- The workstation ran only `py_compile`, `bash -n`, argument-parser help, and
  HTTP clients; it did not import vLLM, start Docker, or execute GPU code.
- The already-running `kiwi` Spark service used development image
  `sha256:930c8cfb8ebdedcbb23528780b83a1d7327578ab11bdfc1a927b1548aa2c1589`
  from recipe commit `fe4749a155d00feda826bd579dd8fae004e26565`.
- vLLM 0.28's installed CLI and parser registry were inspected on that Spark
  and expose `deepseek_v4` tokenizer, tool, and reasoning parsers plus the
  image-limit and generation-config flags now selected by the launch recipe.

## Minimal live checks

- Code-agent client: one warmup and one measured eight-token C1 request, plus
  the same at the depth-zero path, completed with exact streamed token counts.
  The measured samples exposed 80% draft acceptance, proving the Prometheus
  delta and per-sequence stream accounting paths work.
- Prefill client: exact 512-token construction round-tripped through the server
  tokenizer and the server reported exactly 512 prompt tokens.
- Content schema v3: seven of eight semantic arms passed on the first
  one-repeat smoke. The topic arm was truncated by its inherited 224-token
  speed cap after four of five bullets; restoring the Brandon contract's
  384-token budget produced all five bullets and passed the targeted repeat.
- The source-only validator fixtures cover code syntax/type/docstring/asserts,
  arithmetic, fable length/moral, greeting, five-part exposition, both exact
  JSON modes, and Traditional Chinese copy-on-write semantics.

Native Vision overflow rejection, the production tool call, 128K tests, the
full five-run curves, and the C4 soak remain intentionally deferred to the
single frozen production-candidate image.
