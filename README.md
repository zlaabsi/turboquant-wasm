# turboquant-wasm

TurboQuant vector quantization for browser and edge runtimes.

`turboquant-wasm` is a Rust-to-WASM implementation of TurboQuant Algorithm 1 for compressing embedding vectors and searching them client-side. It is designed for local-first search, browser-side RAG, and serverless edge deployments where you want to keep memory use low and avoid a vector database for moderate index sizes.

The GitHub repository is `turboquant-wasm`. The npm package name is `@zlaabsi/turboquant-wasm`.

## Why This Repo Exists

- Compress embeddings to roughly `dim * bits / 8 + 4` bytes per vector.
- Search directly over compressed vectors without full decode.
- Run entirely in the browser, Node.js, Deno, or edge runtimes.
- Persist indexes with `save()` / `load()` for IndexedDB, Cache API, or local files.
- Ship concrete examples instead of only an API surface.

## Quick Start

### Install from npm

```bash
npm install @zlaabsi/turboquant-wasm
```

### Build from source

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

cd turboquant-wasm
npm run build
```

`npm run build` generates the browser package in `pkg/`.
Use `npm run build:node` when you want the Node.js bindings in `pkg-node/`.

### Minimal browser usage

```ts
import { createQuantizer } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.buildIndex(embeddings, nVectors);
const topK = index.search(queryEmbedding, 10);
```

### Persist and reload an index

```ts
import { createQuantizer, Index } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.buildIndex(embeddings, nVectors);

const bytes = index.save();
localStorage.setItem("index", JSON.stringify(Array.from(bytes)));

const restored = new Uint8Array(JSON.parse(localStorage.getItem("index")!));
const reloaded = Index.load(restored, quantizer);
```

## Examples

See [examples/README.md](examples/README.md) for the full matrix and launch instructions.

| Example | Stack | What it shows |
|---|---|---|
| [browser](examples/browser/README.md) | Plain HTML + bag-of-words | Zero-dependency search demo |
| [transformers-js](examples/transformers-js/README.md) | Transformers.js + WebGPU | Real semantic search in-browser |
| [onnx-webgpu](examples/onnx-webgpu/README.md) | ONNX Runtime Web + WebGPU | Quantized ONNX embeddings + IndexedDB |
| [cloudflare](examples/cloudflare/README.md) | Cloudflare Worker | Edge search API with WASM |

## Cookbook

Use-case guides live in [cookbook/README.md](cookbook/README.md).

- [Browser Search](cookbook/browser-search.md)
- [Client-Side RAG](cookbook/client-rag.md)
- [Edge and Serverless](cookbook/edge-serverless.md)
- [Desktop and Mobile](cookbook/desktop-mobile.md)

## Concrete Applications

### Search in the browser

- Static-site search for docs, blogs, or catalogs without a backend.
- Offline PWA search over a product or content index.
- Chrome or Chromium extensions that index tabs locally and search by similarity.

### Client-side RAG

- PDF chat where chunking, embedding, compression, and retrieval all stay on-device.
- Note-taking apps with semantic search over local notes.

### Edge and serverless

- Cloudflare Worker search endpoints with a prebuilt compressed index.
- Vercel Edge Functions, Deno Deploy, or other WASM-friendly runtimes.
- Small embedded devices running a WASM runtime for local retrieval.

### Desktop and mobile

- Electron or Tauri plugins for local file search.
- IDE code search over precomputed code embeddings.
- Mobile delivery where compressed vectors reduce transfer size before client-side search.

## Performance Snapshot

This project is useful enough to demonstrate concrete browser and edge use cases, but it is **not benchmarked enough yet to claim universal superiority or production readiness on every target**.

Current evidence is a single-machine local snapshot on `Apple M1 Max`, `Node v22.11.0`, `npm 10.9.0`, and `Darwin 25.3.0 arm64`, using synthetic clustered embeddings. What is still missing:

- repeated runs with variance reporting for every metric
- a lower-variance harness for build and search sweeps
- browser benchmarks across low-end and mid-range devices
- public real-world embedding corpora
- head-to-head comparisons against exact float32 search and graph-based libraries

The current source of truth is the structured snapshot in [benchmarks/results/2026-04-08-m1-max-node22.json](benchmarks/results/2026-04-08-m1-max-node22.json) plus the raw console log in [benchmarks/results/2026-04-08-m1-max-node22-realworld.txt](benchmarks/results/2026-04-08-m1-max-node22-realworld.txt).

Even with those caveats, the current snapshot does show the implementation is viable for moderate corpus sizes:

| Scenario | Snapshot |
|---|---|
| `d=384`, `4-bit`, `N=5000` | `82.4%` recall@10, `11.89 ms` median search in the clustered-query sweep, `196 B/vector` |
| `d=768`, `4-bit`, `N=3000` | `81.5%` recall@10, `10.37 ms` median search, `388 B/vector` |
| Sustained load at `N=5000` | `7.87 ms` p50, `10.07 ms` p95, `23.13 ms` p99, `112 q/s` |
| Web bundle size | `83.7 KiB` raw, `31.1 KiB` gzip |

![Recall vs bit-width](benchmarks/charts/recall-vs-bits.svg)

![Search latency vs corpus size](benchmarks/charts/search-vs-corpus.svg)

![Tail latency under sustained load](benchmarks/charts/tail-latency-5k.svg)

![Bundle size](benchmarks/charts/bundle-size.svg)

## Development

### Useful commands

```bash
npm run build       # browser bindings
npm run build:node  # node bindings
npm run build:all   # both targets
npm run test        # wasm-bindgen tests in Node
npm run verify      # cargo check + tests + npm pack dry-run
npm run bench       # synthetic node benchmark
npm run bench:realworld
npm run bench:charts
```

If you want a single local entrypoint instead of memorizing scripts:

```bash
make setup
make verify
make serve PORT=8080
```

### Release

1. Update `package.json` if you are cutting a new version.
2. Run `npm run verify`.
3. Push `main`.
4. Create and push a matching tag such as `v0.1.0`.
5. The `release` workflow publishes a GitHub Release and publishes to npm when `NPM_TOKEN` is configured.
6. You can also trigger the workflow manually from GitHub Actions if you need to rerun a tag release.

### Run the examples locally

```bash
npm run build
python3 -m http.server 8080
```

Then open:

- `http://localhost:8080/examples/browser/`
- `http://localhost:8080/examples/transformers-js/`
- `http://localhost:8080/examples/onnx-webgpu/`

## Repository Layout

```text
src/                 Rust core
tests/               wasm-bindgen tests
examples/            runnable demos
cookbook/            architecture guides and integration patterns
benchmarks/          local benchmarking scripts
index.mjs            browser-friendly package entrypoint
index.node.mjs       Node.js package entrypoint
index.cjs            CommonJS package entrypoint
```

## Notes

- The rotation matrix costs `O(dim^2)` memory. `dim=384` is comfortable for browser use. `dim=1536` is still viable, but heavier to initialize.
- This repo implements TurboQuant Algorithm 1 only. QJL is intentionally omitted for the browser-focused memory budget.
- `pkg/` and `pkg-node/` are generated build outputs and are not committed to source control.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- [PolarQuant](https://arxiv.org/abs/2502.02617)
- [QJL](https://arxiv.org/abs/2406.03482)

## License

Apache-2.0
