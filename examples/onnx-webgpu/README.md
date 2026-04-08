# ONNX Runtime WebGPU Example

Semantic search in the browser using ONNX Runtime Web, a quantized MiniLM model, and a lightweight WordPiece tokenizer implementation.

## What it demonstrates

- Explicit ONNX Runtime session setup.
- WebGPU execution provider with fallback.
- Tokenization and mean pooling in userland.
- IndexedDB persistence for the compressed index.

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/onnx-webgpu/`.

## Notes

- This example is useful when you want control over model files, tokenization, and runtime configuration.
- It is a better starting point than the Transformers.js example if you expect to swap models or pre/post-processing steps.
