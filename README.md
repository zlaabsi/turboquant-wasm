# turboquant-wasm

Compress embedding vectors and search them in the browser, offline, or at the edge.

`turboquant-wasm` is a Rust/WASM implementation of TurboQuant Algorithm 1. It is built for applications that already have embeddings and want local retrieval without shipping a vector database or a graph index.

## Why use it

- Compress embeddings aggressively. With `4-bit` quantization, a `384d` vector takes about `196 B` and a `768d` vector about `388 B`.
- Search directly over compressed vectors. No full decode step on every query.
- Persist indexes with `save()` and restore them with `Index.load()`.
- Run in browsers, Node.js, and WASM-friendly edge runtimes.
- Start from runnable examples instead of just a low-level API.

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
