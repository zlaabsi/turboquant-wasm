# turboquant-wasm: Benchmark & Comparative Analysis

> Status: maintained analysis for the current repository.
> TurboQuant rows in this document are refreshed from `benchmarks/results/2026-04-09-m1-max-node22.json`.
> Competitor rows remain estimate-based comparison points for positioning. They are useful for tradeoff discussion, not a fresh side-by-side rerun in this repo.

**Last refreshed:** 2026-04-09  
**Current TurboQuant snapshot:** 2026-04-09 refresh of the 2026-04-08 M1 Max run — Apple M1 Max, Node v22.11.0, npm 10.9.0, Darwin 25.3.0 arm64  
**Paper:** arXiv:2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"

---

## 1. Bundle Size Analysis

### Current measured package

These numbers come directly from the current benchmark snapshot and match the values surfaced in the README.

| File | Size |
|---|---|
| `turboquant_wasm_bg.wasm` | `63,943 bytes` (`62.4 KiB`) |
| `turboquant_wasm.js` + `turboquant_wasm_bg.js` | `18,491 bytes` (`18.1 KiB`) |
| **Total raw** | **`82,434 bytes` (`80.5 KiB`)** |
| **`.wasm` gzip** | **`27,279 bytes` (`26.6 KiB`)** |
| **`js` gzip** | **`3,758 bytes` (`3.7 KiB`)** |
| **Total gzip** | **`31,037 bytes` (`30.3 KiB`)** |

### Comparison with alternative browser-side vector search libraries

TurboQuant values below are current measured numbers. Alternative-library rows are maintained comparison estimates for browser-side positioning.

| Library | `.wasm` gzip | JS glue gzip | Total gzip | Notes |
|---|---|---|---|---|
| **turboquant-wasm** | **`26.6 KiB`** | **`3.7 KiB`** | **`30.3 KiB`** | Quantization-first, no graph index |
| usearch-wasm | `~200 KiB` | `~15 KiB` | `~215 KiB` | HNSW + SIMD |
| Voy | `~150 KiB` | `~20 KiB` | `~170 KiB` | Rust HNSW |
| hnswlib-wasm | `~300 KiB` | `~25 KiB` | `~325 KiB` | C++ via Emscripten |
| vectra | `0 KiB` | `~50 KiB` | `~50 KiB` | Pure JS brute-force |

Current positioning takeaway:

- `turboquant-wasm` is about `7.1x` smaller than `usearch-wasm`.
- `turboquant-wasm` is about `5.6x` smaller than `Voy`.
- `turboquant-wasm` is about `10.7x` smaller than `hnswlib-wasm`.
- `turboquant-wasm` is still about `1.7x` smaller than `vectra` in total shipped bytes.

This size profile is the main differentiator for:

- edge/serverless deployments with tight WASM budgets
- mobile web and embedded widgets where every additional KB hurts
- client-side search features that should not drag a full graph index into the page

### Why it stays small

- No HNSW graph or graph-tuning machinery in the binary.
- No external native dependency stack, BLAS, or LAPACK.
- A small core: PRNG, orthogonalization, centroid tables, scalar quantization, packed storage, and compressed brute-force scan.
- Size-oriented release settings in `Cargo.toml`, plus an implementation that matches the TurboQuant algorithm instead of wrapping a larger ANN engine.
- The published npm browser entrypoint uses the `wasm-pack` bundler target, so the package does not ship a runtime `fetch()`-based Wasm loader.

---

## 2. Feature Comparison

The table below keeps the product-level comparison that matters for repo visitors, but refreshes the TurboQuant numbers to the current implementation.

| Feature | turboquant-wasm | usearch-wasm | Voy | hnswlib-wasm |
|---|---|---|---|---|
| **Bundle size (gzip)** | `30.3 KiB` | `~215 KiB` | `~170 KiB` | `~325 KiB` |
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

### Best fit

- client-side RAG with moderate corpus sizes
- semantic search in SPAs, PWAs, browser extensions, and desktop shells
- edge endpoints that want a tiny deployable package
- privacy-sensitive setups where embeddings and documents stay local

### Not the right fit

- very large corpora where graph ANN is the main requirement
- workloads targeting sub-millisecond latency at high `N`
- benchmark claims that require a committed, reproducible, side-by-side rerun against every alternative

---

## 3. Current Measured Snapshot

This is the current local evidence committed in the repo.

### Environment

- `Apple M1 Max`
- `Node v22.11.0`
- `npm 10.9.0`
- `Darwin 25.3.0 arm64`
- synthetic clustered embeddings

### Snapshot summary

| Scenario | Result |
|---|---|
| `d=384`, `4-bit`, `N=5000` | `82.4%` recall@10, `11.89 ms` median search, `196 B/vector` |
| `d=384`, `4-bit`, `N=10000` | `78.5%` recall@10, `16.09 ms` search |
| `d=768`, `4-bit`, `N=3000` | `81.5%` recall@10, `10.37 ms` median search, `388 B/vector` |
| Sustained load at `N=5000` | `7.87 ms` p50, `10.07 ms` p95, `23.13 ms` p99, `112 q/s` |
| Web package size | `80.5 KiB` raw, `30.3 KiB` gzip |

### Important limits on this snapshot

- single machine only
- Node.js benchmark, not a browser matrix
- synthetic clustered embeddings, not a public benchmark corpus
- no committed side-by-side rerun against usearch, Voy, hnswlib, or exact float32 search

The charts in `benchmarks/charts/` are generated directly from this snapshot plus the maintained comparison table in this document.

---

## 4. Why Current Measured Latencies Are Worse Than the March Estimates

The short answer is that the old March analysis mixed several kinds of numbers:

- exact and estimated bundle-size values
- theoretical FLOP-based latency estimates
- implementation notes from before bit-packing and streaming insertion landed
- simplified assumptions about browser/WASM execution cost

So the March document was never a pure measured benchmark report. It was part sizing note, part theory note, part rough planning model.

### Where the old March analysis was too optimistic

| Metric | March analysis | Current reality | Why they differ |
|---|---|---|---|
| Search total, `d=384`, `N=10000` | `~4.3 ms` estimated | `16.09 ms` measured | The March figure was a FLOP-based estimate, not a real measurement |
| Search total, `d=1536`, `N=10000` | `~20 ms` estimated | no committed measured rerun at that exact point | The old number was still theoretical and should not have been read as a benchmark |

### Main reasons for the gap

- The March search numbers were back-of-the-envelope throughput estimates, not timings captured from the shipping Node/WASM package.
- The current implementation uses bit-packed storage. That cuts memory dramatically, but search now pays real unpacking cost per coordinate during scan.
- The old model mostly counted arithmetic work. Real search time also includes memory traffic, centroid lookup overhead, packed-code extraction, and top-k selection/sorting.
- The benchmark harness measures actual end-to-end `index.search(...)` through the wasm-bindgen surface, not just an idealized inner loop.
- Runtime behavior in Node/V8 with a real WASM module does not match a simple “1 GFLOP/s scalar” assumption closely enough to publish as a final latency claim.

### What improved versus March

Not everything got worse. Some parts improved substantially:

| Metric | March analysis | Current reality | Why it improved |
|---|---|---|---|
| Memory per vector, `d=384`, `4-bit` | `388 B` | `196 B` | Bit-packing is now implemented |
| Memory per vector, `d=1536`, `4-bit` | `1,540 B` | `772 B` | Bit-packing is now implemented |
| Streaming add | Planned | Implemented | Incremental insertion landed |
| Build index, `d=384`, `N=10000` | `~3 s` estimated | `418 ms` measured | The old estimate was rough and current code benefits from the actual implementation path, including SIMD-enabled rotation |

If by "scores" you meant recall rather than latency, the March document did not actually publish higher measured recall@10 values. It mostly cited theoretical distortion/MSE expectations from the paper, which are related to quantization quality but are not the same metric as measured retrieval recall.

---

## 5. Key Advantages Summary

| Dimension | turboquant-wasm advantage |
|---|---|
| **Bundle size** | About `5.6x` to `10.7x` smaller than graph-based WASM alternatives in the maintained comparison table, while still smaller than `vectra` in total shipped bytes |
| **Memory per vector** | About `8x` lower than raw float32 storage at packed `4-bit`, which directly helps browser and edge memory budgets |
| **API simplicity** | Build and search without graph parameters, `ef` tuning, connectivity tuning, or separate quantization passes |
| **Theoretical foundation** | Based on the TurboQuant paper rather than a purely heuristic compression layer |
| **Edge compatibility** | Small enough to fit comfortably in edge/serverless WASM budgets, including Cloudflare Worker-style deployments |
| **No training** | Centroids are fixed and quantizer creation is deterministic from the chosen seed |
| **Determinism** | Same seed and same inputs produce the same rotation and the same compressed representation |

---

## 6. Trade-offs

| Dimension | Limitation |
|---|---|
| **Search speed at scale** | Search still scales linearly in `N`, so graph ANN wins first on very large corpora |
| **Packed scan cost** | Bit-packing saves memory but adds unpacking work during search |
| **Snapshot breadth** | Current benchmark evidence is still a single-machine snapshot, not a broad device/browser matrix |
| **Competitor comparisons** | Alternative-library rows are maintained estimates, not a committed fresh rerun in this repo |
| **Quantizer init at high dimension** | Quantizer creation remains more noticeable as `dim` grows because the rotation setup is not free |

---

## 7. Reproducing the Current Snapshot

```bash
npm run build:web
npm run build:node
npm run bench:realworld
npm run bench:charts
```

Artifacts:

- measured snapshot: `benchmarks/results/2026-04-09-m1-max-node22.json`
- raw log: `benchmarks/results/2026-04-08-m1-max-node22-realworld.txt`
- comparison data for charts: `benchmarks/results/analysis-derived-comparison.json`
- generated charts: `benchmarks/charts/*.svg`
