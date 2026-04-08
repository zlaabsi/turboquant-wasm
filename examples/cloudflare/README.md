# TurboQuant Cloudflare Worker

This example shows the **deployment pattern**, not the best retrieval quality.

It runs TurboQuant inside a Cloudflare Worker and exposes a small `/search` HTTP API. The vectors are bag-of-words vectors built from a fixed in-worker corpus, so the point here is edge packaging and request flow, not semantic search relevance.

## What it demonstrates

- loading TurboQuant WASM in a Worker
- building a tiny in-memory index once per worker instance
- serving search results through a simple JSON API
- keeping retrieval logic in the edge runtime with no external vector database

## Run locally

Prerequisites:

- Rust + `wasm-pack`
- Node.js 18+
- Wrangler

From the repo root:

```bash
npm run build
cd examples/cloudflare
wrangler dev
```

Open `http://localhost:8787/` or query the API directly:

```bash
curl "http://localhost:8787/search?q=finding+similar+vectors"
```

## Deploy

```bash
cd examples/cloudflare
wrangler login
wrangler deploy
```

## Runtime flow

1. The worker receives a request.
2. `ensureIndex()` lazily initializes TurboQuant once per worker instance.
3. The example builds a fixed bag-of-words index from the hardcoded corpus.
4. `/search?q=...` maps the query to the same vector space and runs `index.search(...)`.
5. The worker returns ids, text, matched words, and latency.

## Where to edit first

- `CORPUS` if you want a different dataset
- the fixed vocabulary if you want different lexical coverage
- `DIM` if the vocabulary changes
- `handleSearch()` if you want a different response schema

## Recommended production path

Do **not** treat this exact demo as the final architecture for semantic search.

For a real edge service:

- precompute embeddings offline
- compress them ahead of time
- load a saved TurboQuant index blob on cold start
- keep metadata in KV, R2, D1, or another store outside the vector index
- return ids and scores, then hydrate metadata separately

That keeps the worker small and makes index updates explicit.

## Why this example still matters

Even with a toy vectorizer, it proves something useful: the retrieval layer itself can live comfortably inside a Worker and answer similarity queries without a dedicated vector service.

For the broader deployment recipe, see [Edge and Serverless](../../cookbook/edge-serverless.md).
