# Examples

These examples are meant to answer "how would I actually wire TurboQuant into an app?" They are references, not polished end-user products.

## Shared setup

```bash
npm run build
python3 -m http.server 8080
```

## Which example should you start from?

| Example | Open at | Embeddings happen where? | Persistence | Edit this first | Use it when | Do not start here if |
|---|---|---|---|---|---|---|
| [browser](browser/README.md) | `http://localhost:8080/examples/browser/` | Nowhere. It uses bag-of-words vectors built in-page. | None | Corpus textarea and `BITS` in `index.html` | You want to understand the API surface, streaming insertion, or a zero-dependency smoke test | You need real semantic search quality |
| [transformers-js](transformers-js/README.md) | `http://localhost:8080/examples/transformers-js/` | In the browser via Transformers.js | IndexedDB | Corpus textarea, model call, IndexedDB keys | You want the fastest path to "real embeddings + compressed local search" in one page | You need explicit control over tokenizer/model files |
| [onnx-webgpu](onnx-webgpu/README.md) | `http://localhost:8080/examples/onnx-webgpu/` | In the browser via ONNX Runtime Web | IndexedDB | `MODEL_URL`, `TOKENIZER_URL`, `DIM`, `MAX_SEQ_LEN` | You want full control over model assets, tokenization, and runtime settings | You just want the shortest demo to adapt |
| [cloudflare](cloudflare/README.md) | `wrangler dev` | Nowhere. It uses bag-of-words vectors inside the worker. | In-memory worker state | `CORPUS` and vocabulary in `worker.js` | You want an edge deployment pattern and HTTP search API | You need a production semantic search backend immediately |

## What each example actually proves

- `browser/` proves the TurboQuant WASM API, including `new_empty`, `add_vector`, and `search`, without hiding anything behind a model stack.
- `transformers-js/` proves you can run local embedding inference, compress the resulting vectors, persist the index, and search it later in a normal browser tab.
- `onnx-webgpu/` proves the same end-to-end flow with explicit model URLs, tokenizer loading, mean pooling, and execution-provider control.
- `cloudflare/` proves the retrieval layer itself works inside an edge worker and can sit behind a minimal HTTP API.

## What is not covered yet

The repo still does **not** ship dedicated examples for:

- PDF ingestion and chunking
- browser extensions
- prebuilt index download from static hosting
- Electron or Tauri packaging
- React Native or Hermes integration

Those patterns are covered in the cookbook:

- [Browser Search](../cookbook/browser-search.md)
- [Browser Extension](../cookbook/browser-extension.md)
- [Client-Side RAG](../cookbook/client-rag.md)
- [Edge and Serverless](../cookbook/edge-serverless.md)
- [Desktop and Mobile](../cookbook/desktop-mobile.md)
