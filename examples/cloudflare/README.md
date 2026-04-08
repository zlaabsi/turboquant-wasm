# TurboQuant Cloudflare Worker

Vector search at the edge using TurboQuant WASM. Indexes ~50 sentences with
4-bit quantization and serves search queries via a simple HTTP API.

## Prerequisites

- [Rust](https://rustup.rs/) and [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/) (v18+)
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/)

## Setup

```bash
# Install wrangler (if not already installed)
npm install -g wrangler

# Build the WASM module (from turboquant-wasm root)
cd ../..
wasm-pack build --target web --out-dir pkg

# Return to the example directory
cd examples/cloudflare
```

## Local development

```bash
wrangler dev
```

Open http://localhost:8787 for the search UI, or query directly:

```bash
curl "http://localhost:8787/search?q=finding+similar+vectors"
```

## Deploy

```bash
wrangler login   # one-time auth
wrangler deploy
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | HTML search form |
| `GET /search?q=...` | JSON search results (top 5) |

### Example response

```json
{
  "query": "vector similarity",
  "results": [
    { "rank": 1, "index": 11, "text": "Cosine similarity measures the angle between two vectors", "score": 0.4472, "matched_words": ["similarity", "vectors"] },
    { "rank": 2, "index": 2, "text": "Vector search finds similar items in a database", "score": 0.3162, "matched_words": ["vector"] }
  ],
  "search_ms": 0.12,
  "n_vectors": 50,
  "memory_bytes": 5200,
  "vocabulary_size": 200,
  "bits": 4
}
```

## How it works

1. A fixed vocabulary of ~200 common words defines the embedding dimension.
2. Each sentence is mapped to a bag-of-words vector (word counts, L2-normalized).
3. Vectors are compressed with TurboQuant Algorithm 1 (4-bit, ~8x compression).
4. At search time, the query is embedded the same way and searched against the
   compressed index. All computation runs inside the Worker via WASM.

Real applications would use neural embedding models (e.g., from the
Sentence Transformers library) for much better semantic matching. This demo
shows the TurboQuant WASM API running entirely at the edge.
