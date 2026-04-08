# Edge and Serverless

Use this pattern when you want a small search service at the edge without a separate vector database.

Typical fits:

- Cloudflare Worker search APIs
- edge functions serving a prebuilt local index
- small WASM-friendly runtimes where embeddings are prepared elsewhere

## The important distinction

There are two very different deployment patterns:

### Demo pattern

Build the index inside the runtime from a tiny fixed corpus on cold start.

This is what the repo's Cloudflare example does. It is good for showing the runtime shape, but not the recommended production default.

### Production pattern

Precompute embeddings and TurboQuant bytes offline, then load the saved index blob inside the edge runtime.

That avoids spending cold-start time on corpus embedding or index construction.

## Recommended production architecture

```text
offline job -> embeddings -> TurboQuant save() -> index blob
                                            |
request -> edge runtime -> load blob once -> search -> ids/scores -> metadata hydration
```

## Offline build example

```ts
import fs from "node:fs/promises";
import { createQuantizer, flattenEmbeddings } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const flat = flattenEmbeddings(items.map((item) => item.embedding));
const index = quantizer.buildIndex(flat, items.length);

await fs.writeFile("dist/search.index.bin", index.save());
await fs.writeFile("dist/search.meta.json", JSON.stringify(items.map(({ id, slug, title }) => ({ id, slug, title }))));
```

## Runtime search example

```ts
let cachedIndex = null;
let cachedMeta = null;
let quantizer = null;

async function ensureIndex(env) {
  if (cachedIndex) return;
  quantizer = await createQuantizer({ dim: 384, bits: 4 });
  const bytes = new Uint8Array(await env.SEARCH_INDEX.arrayBuffer());
  cachedIndex = Index.load(bytes, quantizer);
  cachedMeta = await env.SEARCH_META.json();
}

async function search(env, queryEmbedding) {
  await ensureIndex(env);
  const ids = cachedIndex.search(queryEmbedding, 5);
  return ids.map((i) => cachedMeta[i]);
}
```

## Query embedding options

- lexical or bag-of-words mapping in the runtime for tiny demos
- external embedding service before the edge function if privacy is not strict
- lightweight local embedder only if the runtime can actually support it

In most serverless setups, only retrieval should happen in the edge runtime.

## Storage guidance

- Keep vector bytes in a single blob or asset.
- Keep metadata in a separate JSON file, KV namespace, object store, or DB table.
- Keep the worker response schema stable around ids and scores.

## Good fits

- corpora that change by deployment, not by request
- "small enough to fit comfortably in memory" indexes
- edge latency where a graph database would be operationally overkill

## Common mistakes

- building the index on every request
- bundling too much metadata into the worker itself
- trying to do expensive query embedding in a constrained edge runtime
- assuming the Cloudflare demo corpus/vectorizer is representative of production relevance

## Best matching repo resource

- [examples/cloudflare](../examples/cloudflare/README.md)
