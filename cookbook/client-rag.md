# Client-Side RAG

## Good fits

- PDF chat where documents must stay local.
- Knowledge bases inside a desktop-like web app.
- Note-taking apps with private semantic retrieval.

## Recommended architecture

1. Parse files locally into chunks.
2. Embed chunks in-browser.
3. Compress and index with TurboQuant.
4. Persist the compressed index in IndexedDB.
5. At query time, embed the prompt and retrieve top chunks locally.

## Pattern

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.buildIndex(chunkEmbeddings, chunks.length);

const saved = index.save();
await putIntoIndexedDb("notes-index-v1", saved);

const restored = await getFromIndexedDb("notes-index-v1");
const reloaded = Index.load(restored, quantizer);
const topK = reloaded.search(queryEmbedding, 6);
```

## What TurboQuant is responsible for

- Compressing chunk embeddings.
- Storing a compact searchable index.
- Returning ids of relevant chunks.

## What stays outside TurboQuant

- PDF parsing.
- Chunking strategy.
- Embedding model selection.
- Final prompt assembly for the LLM.

## Practical starting points

- [examples/transformers-js](../examples/transformers-js/README.md)
- [examples/onnx-webgpu](../examples/onnx-webgpu/README.md)
