# Transformers.js Example

This is the shortest serious example in the repo: real embeddings in the browser, 4-bit compression, local search, and IndexedDB persistence.

It uses `Xenova/all-MiniLM-L6-v2` through Transformers.js, prefers WebGPU when available, and falls back to the Transformers.js WASM backend if needed.

## What it demonstrates

- loading TurboQuant WASM and a browser embedding pipeline in the same page
- generating `384d` normalized embeddings in-browser
- compressing them with `TurboQuantizer(dim=384, bits=4)`
- persisting the compressed index with `save()`
- restoring it later with `CompressedIndex.load(...)`

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/transformers-js/`.

## Runtime flow

1. Load the TurboQuant WASM module.
2. Probe `navigator.gpu` and attempt a WebGPU embedding backend.
3. Load `Xenova/all-MiniLM-L6-v2` through the Transformers.js `pipeline(...)`.
4. Read the corpus from the textarea.
5. Embed each sentence into a `384d` vector.
6. Build a compressed index with `build_index(...)`.
7. Save `{ index bytes, sentences }` into IndexedDB.
8. At query time, embed the query, search the compressed index, and render the top ids.

## Where to edit first

- `const DIM = 384`, `const BITS = 4`, `const TOP_K = 5`
- `DB_NAME` and `DB_STORE` if you need versioned persistence
- the corpus textarea content
- the `pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', ...)` call if you want another Transformers.js model

## What is expensive here

The expensive part is almost always embedding, not TurboQuant search.

- first load downloads model assets
- building the index is dominated by embedding the corpus
- search latency is split into query embedding plus compressed-vector search

That is why this demo is useful: it shows the actual cost breakdown users will feel in a browser tab.

## Good production follow-ups

- precompute embeddings offline for mostly-static corpora
- keep only metadata and compressed index in the client bundle or IndexedDB
- move corpus embedding off the critical path if startup latency matters
- separate document metadata from the vector index instead of storing everything in one IndexedDB record
- version the stored index by model, chunking strategy, and corpus revision

## Important caveats

- The demo re-embeds result sentences to display cosine scores. That is fine for a demo, but not ideal in production.
- The corpus lives in a textarea. Real apps should treat it as content input, not storage.
- There is no chunking pipeline here. If you want RAG over long documents, see [Client-Side RAG](../../cookbook/client-rag.md).

## When to choose this over the ONNX example

Choose this example when:

- you want the shortest browser semantic-search prototype
- you are okay with the Transformers.js abstraction layer
- you do not need to own tokenization and low-level runtime config

Choose [onnx-webgpu](../onnx-webgpu/README.md) when you want explicit control over the model files, tokenizer, and ORT execution provider.
