# Transformers.js Example

Real semantic search in the browser using `Xenova/all-MiniLM-L6-v2` with WebGPU when available and WASM fallback otherwise.

## What it demonstrates

- In-browser embedding generation with Transformers.js.
- TurboQuant compression at 4 bits per coordinate.
- Search over a compressed index.
- IndexedDB persistence via `save()` and `CompressedIndex.load()`.

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/transformers-js/`.

## Notes

- First load downloads the embedding model.
- Browsers with WebGPU support will use it automatically when the pipeline succeeds.
- If WebGPU is unavailable or fails, the demo falls back to the Transformers.js WASM backend.
