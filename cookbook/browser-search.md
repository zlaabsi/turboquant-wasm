# Browser Search

Use this pattern when your app runs mostly in the browser and you want local retrieval without standing up a vector database.

Typical fits:

- docs search on a static site
- blog or catalog search in a SPA
- offline PWA search
- "download once, search locally" product indexes

## Recommended default architecture

For mostly-static corpora, do **not** embed and compress everything in the browser on every visit.

Recommended path:

1. Generate embeddings offline or at build time.
2. Compress them into a TurboQuant index once.
3. Ship `search.index.bin` and `search.meta.json` with the site.
4. Load both in the browser on startup.
5. Embed only the query in the browser.
6. Search locally and map ids back to metadata.

## Data layout

Keep vectors and content metadata separate:

- `search.index.bin`: output of `index.save()`
- `search.meta.json`: `{ id, title, url, snippet, tags }[]`
- optional `search.version.json`: model version, index version, corpus revision

That makes cache invalidation and partial UI hydration much simpler.

## Build-time compression example

```ts
import fs from "node:fs/promises";
import { createQuantizer, flattenEmbeddings } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const flat = flattenEmbeddings(documents.map((doc) => doc.embedding));
const index = quantizer.buildIndex(flat, documents.length);

await fs.writeFile("public/search.index.bin", index.save());
await fs.writeFile(
  "public/search.meta.json",
  JSON.stringify(documents.map(({ id, title, url, snippet }) => ({ id, title, url, snippet })))
);
```

## Browser load + search example

```ts
import { createQuantizer, Index } from "@zlaabsi/turboquant-wasm";

const quantizer = await createQuantizer({ dim: 384, bits: 4 });

const [indexBytes, meta] = await Promise.all([
  fetch("/search.index.bin").then((r) => r.arrayBuffer()),
  fetch("/search.meta.json").then((r) => r.json()),
]);

const index = Index.load(new Uint8Array(indexBytes), quantizer);

const queryEmbedding = await embed(queryText);
const ids = index.search(queryEmbedding, 8);
const hits = ids.map((i) => meta[i]);
```

## When to build in-browser instead

Build the index in the browser only when:

- the corpus is user-generated
- the corpus changes per user
- offline-first behavior matters more than first-run latency

That is the path shown in [examples/transformers-js](../examples/transformers-js/README.md) and [examples/onnx-webgpu](../examples/onnx-webgpu/README.md).

## Practical decisions

- `4-bit` is the default starting point for `384d` and `768d` embeddings.
- Keep snippets, titles, tags, and URLs out of the vector blob.
- Version by `(model, bits, corpus revision)` so cached indexes are safe to reuse.
- If startup matters, fetch the compressed index eagerly and delay query-model loading until first interaction.

## Common mistakes

- shipping raw embeddings when the corpus is static
- rebuilding the index on every page load
- storing all metadata inside IndexedDB along with the vector bytes with no version key
- assuming lexical and semantic search should use the same ranking pipeline

## Best matching repo resources

- API sanity check: [examples/browser](../examples/browser/README.md)
- Real in-browser embeddings: [examples/transformers-js](../examples/transformers-js/README.md)
- Lower-level browser inference: [examples/onnx-webgpu](../examples/onnx-webgpu/README.md)
