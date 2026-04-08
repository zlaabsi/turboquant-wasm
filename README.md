<p align="center">
  <img src="docs/assets/turboquant-wasm.png" alt="TurboQuant WASM" width="947">
</p>

<p align="center">
  <strong>Training-free embedding compression and local vector search for browsers, offline apps, and edge runtimes.</strong>
</p>

<p align="center">
  <img src="benchmarks/charts/bundle-size-vs-alternatives.svg" alt="Bundle size vs alternative browser-side search libraries" width="860">
</p>

<p align="center">
  Fast read: the purple bar is the current <code>turboquant-wasm</code> build at about <code>31.1 KiB</code> gzip. Gray bars are maintained comparison estimates from <code>benchmarks/wasm_analysis.md</code>, and the small labels underneath show how much larger they are relative to TurboQuant. This is positioning context, not a committed side-by-side rerun against every alternative.
</p>

`turboquant-wasm` is a Rust/WebAssembly implementation of the TurboQuant MSE variant (Algorithm 1 from the paper). It is built for applications that already have embeddings and want local retrieval without shipping a vector database or a graph index.

## Why this repo does not ship the QJL variant

The short version is that QJL works against the main design goal of `turboquant-wasm`: keep browser-side retrieval small and memory-efficient.

- QJL adds an extra projection matrix, which materially increases runtime memory pressure.
- In browser and WASM settings, that extra matrix becomes expensive quickly, especially once embedding dimensions get large.
- The MSE variant already gives strong recall for the bit-rates this repo actually targets in practice, especially at `3+` bits.
- For this project, the tradeoff was not worth it: more complexity and more memory, without fitting the core promise of a tiny browser-first package.

So the repo deliberately optimizes for the TurboQuant MSE path: smaller package, lower memory footprint, simpler runtime story.

## At a glance

- Small web package. The current measured build is about `31.1 KiB` gzip.
- Aggressive compression. With `4-bit` quantization, a `384d` vector takes about `196 B` and a `768d` vector about `388 B`.
- Direct search on compressed vectors. No full decode step on every query.
- Portable packaging. Runs in browsers, Node.js, and WASM-friendly edge runtimes.
- Persistence built in. Save indexes with `save()` and restore them with `Index.load()`.
- Example-first repo. Includes browser, WebGPU, and Cloudflare demos.

## Bundle Size Analysis

Current `turboquant-wasm` bundle numbers below come from the latest measured snapshot in `benchmarks/results/2026-04-08-m1-max-node22.json`. Alternative-library rows are maintained comparison estimates from `benchmarks/wasm_analysis.md`, not a fresh side-by-side rerun in this repo.

### Current measured package

| File | Size |
|---|---|
| `turboquant_wasm_bg.wasm` | `63,943 bytes` (`62.4 KiB`) |
| `turboquant_wasm.js` | `21,768 bytes` (`21.3 KiB`) |
| **Total raw** | **`85,711 bytes` (`83.7 KiB`)** |
| **`.wasm` gzip** | **`27,372 bytes` (`26.7 KiB`)** |
| **`js` gzip** | **`4,466 bytes` (`4.4 KiB`)** |
| **Total gzip** | **`31,838 bytes` (`31.1 KiB`)** |

### Comparison with alternative browser-side vector search libraries

| Library | `.wasm` gzip | JS glue gzip | Total gzip | Notes |
|---|---|---|---|---|
| **turboquant-wasm** | **`26.7 KiB`** | **`4.4 KiB`** | **`31.1 KiB`** | Quantization-first, no graph index |
| usearch-wasm | `~200 KiB` | `~15 KiB` | `~215 KiB` | HNSW + SIMD |
| Voy | `~150 KiB` | `~20 KiB` | `~170 KiB` | Rust HNSW |
| hnswlib-wasm | `~300 KiB` | `~25 KiB` | `~325 KiB` | C++ via Emscripten |
| vectra | `0 KiB` | `~50 KiB` | `~50 KiB` | Pure JS brute-force |

`turboquant-wasm` is materially smaller than graph-based WASM alternatives. That matters most for edge deployments, mobile web, and embedded search widgets where bundle budget is tight.

### Why it stays small

- No HNSW graph or graph-tuning machinery in the binary.
- No external native dependency stack, BLAS, or LAPACK.
- A small core: PRNG, orthogonalization, centroid tables, scalar quantization, packed storage, and compressed brute-force scan.
- Size-oriented WASM build settings, plus a design that matches the algorithm instead of wrapping a larger ANN engine.

## Feature Comparison

This table keeps the product-level comparison from `benchmarks/wasm_analysis.md`, but refreshes the `turboquant-wasm` numbers to the current implementation.

| Feature | turboquant-wasm | usearch-wasm | Voy | hnswlib-wasm |
|---|---|---|---|---|
| **Bundle size (gzip)** | `31.1 KiB` | `~215 KiB` | `~170 KiB` | `~325 KiB` |
| **Training needed** | No | No, but graph build required | No, but graph build required | No, but graph build required |
| **Quantization** | `1-8` bit scalar, paper-backed | `8-bit` scalar | None | None |
| **Search algorithm** | Brute-force scan in rotated domain | HNSW graph | HNSW graph | HNSW graph |
| **Search complexity** | `O(Nd)` | `O(log N * d)` | `O(log N * d)` | `O(log N * d)` |
| **Memory per vector (`d=384`, `4-bit`)** | `196 B` | `1,536 B` | `1,536 B` | `1,536 B` |
| **Memory per vector (`d=1536`, `4-bit`)** | `772 B` | `6,144 B` | `6,144 B` | `6,144 B` |
| **Compression ratio** | `~8x` at packed `4-bit` | `1x` | `1x` | `1x` |
| **Index build** | `O(N * d^2)` | `O(N * log N * d)` | `O(N * log N * d)` | `O(N * log N * d)` |
| **Browser support** | All modern browsers with WASM | Browser WASM targets | Browser WASM targets | Browser WASM targets |
| **Streaming add** | Yes | Yes | Yes | Yes |
| **Theoretical guarantees** | MSE-optimal quantization from TurboQuant | None | None | None |
| **Paper-backed approach** | Yes | No | No | No |

## Key Advantages Summary

| Dimension | turboquant-wasm advantage |
|---|---|
| **Bundle size** | About `5.5x` to `10.5x` smaller than graph-based WASM alternatives in the comparison tables, while still smaller than `vectra` in total shipped bytes |
| **Memory per vector** | About `8x` lower than raw float32 storage at packed `4-bit`, which matters directly in browser and edge memory budgets |
| **API simplicity** | Build and search without graph parameters, `ef` tuning, connectivity tuning, or external quantization passes |
| **Theoretical foundation** | Based on the TurboQuant paper rather than a purely heuristic compression layer |
| **Edge compatibility** | Small enough to fit comfortably in edge/serverless WASM budgets, including Cloudflare Worker-style deployments |
| **No training** | Centroids are fixed and quantizer creation is deterministic from the chosen seed |
| **Determinism** | Same seed and same inputs produce the same rotation and the same compressed representation |

## Good fit

- Static-site search for docs, blogs, and catalogs
- Local-first semantic search in PWAs or desktop apps
- Client-side RAG where documents never leave the machine
- Browser extensions indexing tabs or notes locally
- Edge APIs with a prebuilt compressed index

## Probably not the right tool

- Very large corpora where you want graph-based ANN over `100k+` vectors
- Workloads that need sub-millisecond latency at large `N`
- Benchmarks where you need a mature head-to-head comparison suite today

## Install

```bash
npm install @zlaabsi/turboquant-wasm
```

## Quick start

### Minimal usage

```ts
import { createQuantizer } from "@zlaabsi/turboquant-wasm";

const dim = 384;
const bits = 4;

const quantizer = await createQuantizer({ dim, bits });
const index = quantizer.buildIndex(embeddings, nVectors);
const resultIds = index.search(queryEmbedding, 10);
```

### Persist and reload

```ts
import { createQuantizer, Index } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.buildIndex(embeddings, nVectors);

const bytes = index.save();
const restored = Index.load(bytes, quantizer);
const resultIds = restored.search(queryEmbedding, 10);
```

### Build from source

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

git clone https://github.com/zlaabsi/turboquant-wasm.git
cd turboquant-wasm
npm run build
```

Use `npm run build:node` when you also want the Node.js target in `pkg-node/`.

## Try the examples

```bash
npm run build
python3 -m http.server 8080
```

Then open:

- `http://localhost:8080/examples/browser/`
- `http://localhost:8080/examples/transformers-js/`
- `http://localhost:8080/examples/onnx-webgpu/`

Example matrix:

| Example | Stack | Best for |
|---|---|---|
| [browser](examples/browser/README.md) | Plain HTML + bag-of-words | Zero-dependency smoke test |
| [transformers-js](examples/transformers-js/README.md) | Transformers.js + WebGPU | Fastest path to real semantic search in-browser |
| [onnx-webgpu](examples/onnx-webgpu/README.md) | ONNX Runtime Web + WebGPU | More control over model and tokenizer |
| [cloudflare](examples/cloudflare/README.md) | Cloudflare Worker | Edge search API pattern |

More detail: [examples/README.md](examples/README.md)

## Cookbook

Use these guides when you want an integration pattern instead of a toy demo:

- [Browser Search](cookbook/browser-search.md)
- [Browser Extension](cookbook/browser-extension.md)
- [Client-Side RAG](cookbook/client-rag.md)
- [Edge and Serverless](cookbook/edge-serverless.md)
- [Desktop and Mobile](cookbook/desktop-mobile.md)

## Performance snapshot

Honest version: the implementation looks useful for moderate corpus sizes, but this repo still does **not** have a full benchmark suite across devices, browsers, public datasets, and competing libraries.

The table below is the current source of truth for measured TurboQuant behavior in this repo. The old March analysis mixed theory, estimates, and older implementation assumptions; `benchmarks/wasm_analysis.md` now explains explicitly why current measured search latency is higher than those early estimates.

Current evidence is a local snapshot on:

- `Apple M1 Max`
- `Node v22.11.0`
- `npm 10.9.0`
- `Darwin 25.3.0 arm64`
- synthetic clustered embeddings

That means the numbers below are **directional evidence**, not a universal SLA.

### Current snapshot

| Scenario | Result |
|---|---|
| `d=384`, `4-bit`, `N=5000` | `82.4%` recall@10, `11.89 ms` median search in the clustered-query sweep, `196 B/vector` |
| `d=768`, `4-bit`, `N=3000` | `81.5%` recall@10, `10.37 ms` median search, `388 B/vector` |
| Sustained load at `N=5000` | `7.87 ms` p50, `10.07 ms` p95, `23.13 ms` p99, `112 q/s` |
| Web package size | `83.7 KiB` raw, `31.1 KiB` gzip |

### Charts

![Recall vs bit-width](benchmarks/charts/recall-vs-bits.svg)

![Search latency vs corpus size](benchmarks/charts/search-vs-corpus.svg)

![Tail latency under sustained load](benchmarks/charts/tail-latency-5k.svg)

![Bundle size](benchmarks/charts/bundle-size.svg)

### Raw benchmark data

- Snapshot JSON: [benchmarks/results/2026-04-08-m1-max-node22.json](benchmarks/results/2026-04-08-m1-max-node22.json)
- Raw console log: [benchmarks/results/2026-04-08-m1-max-node22-realworld.txt](benchmarks/results/2026-04-08-m1-max-node22-realworld.txt)
- Chart generator: [benchmarks/render_charts.js](benchmarks/render_charts.js)

### Comparative context

The charts above are about `turboquant-wasm` alone. The charts below add comparative context using the positioning tables in [benchmarks/wasm_analysis.md](benchmarks/wasm_analysis.md).

Important caveat: these comparative plots are **not** a fresh controlled benchmark suite run side-by-side in this repo. The TurboQuant bars use the current measured package size and current packed storage model; the alternative-library bars come from the maintained comparison estimates in `benchmarks/wasm_analysis.md`. They are here for positioning and tradeoff discussion, not to pretend we already have airtight head-to-head numbers.

Reading guide: purple is the current measured `turboquant-wasm` result, gray bars are the comparison points documented in `benchmarks/wasm_analysis.md`, and the small labels under the gray bars show the relative overhead versus TurboQuant.

![Bundle size vs alternatives](benchmarks/charts/bundle-size-vs-alternatives.svg)

![Memory per vector at d=384](benchmarks/charts/memory-d384-vs-alternatives.svg)

![Memory per vector at d=1536](benchmarks/charts/memory-d1536-vs-alternatives.svg)

### What is still missing

- repeated runs with variance reporting
- lower-variance harnesses for build and search sweeps
- browser benchmarks on low-end and mid-range hardware
- public real-world embedding corpora
- head-to-head comparisons against exact float32 search and graph-based ANN libraries

## API and package notes

- Install from npm with `@zlaabsi/turboquant-wasm`
- Repository: `github.com/zlaabsi/turboquant-wasm`
- Primary workflow: create quantizer -> build or stream index -> save/load -> search
- Generated artifacts live in `pkg/` and `pkg-node/`

## Development

For local workflow, release process, and commit conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).

Common commands:

```bash
npm run build
npm run build:node
npm run test
npm run verify
npm run bench:realworld
npm run bench:charts
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [PolarQuant](https://arxiv.org/abs/2502.02617)
- [QJL](https://arxiv.org/abs/2406.03482)

## License

Apache-2.0
