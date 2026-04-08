# Examples

All examples are static and can be served from the repo root after building the browser target.

## Shared setup

```bash
npm run build
python3 -m http.server 8080
```

## Example matrix

| Example | URL | Embedding stack | Persistence | Best for |
|---|---|---|---|---|
| [browser](browser/README.md) | `http://localhost:8080/examples/browser/` | Bag-of-words | None | API smoke test, tiny demo, zero deps |
| [transformers-js](transformers-js/README.md) | `http://localhost:8080/examples/transformers-js/` | Transformers.js + WebGPU or WASM fallback | IndexedDB | Real semantic search with minimal setup |
| [onnx-webgpu](onnx-webgpu/README.md) | `http://localhost:8080/examples/onnx-webgpu/` | ONNX Runtime Web + WebGPU or WASM EP | IndexedDB | Lower-level control over model + tokenizer |
| [cloudflare](cloudflare/README.md) | `wrangler dev` | Fixed bag-of-words corpus | In-memory worker state | Edge API and deployment pattern |

## Choosing an example

- Start with `browser/` if you only want to verify compression and search.
- Use `transformers-js/` if you want the fastest route to real semantic search in a browser tab.
- Use `onnx-webgpu/` if you want explicit ONNX Runtime control and a tokenizer you can customize.
- Use `cloudflare/` if you want to expose search behind an HTTP API at the edge.
