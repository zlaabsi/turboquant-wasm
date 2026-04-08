# ONNX Runtime WebGPU Example

This is the lower-level browser example. It runs a quantized MiniLM ONNX model with ONNX Runtime Web, implements tokenization and mean pooling explicitly in the page, then compresses the resulting embeddings with TurboQuant.

## Why this example exists

Use this example when the Transformers.js demo is too high-level for your needs.

It shows how to control:

- the model URL
- the tokenizer URL
- the ONNX Runtime execution provider
- tokenization behavior
- mean pooling and normalization
- IndexedDB persistence format

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/onnx-webgpu/`.

## Runtime flow

1. Load TurboQuant WASM.
2. Fetch `tokenizer.json` and build a minimal WordPiece tokenizer.
3. Load the ONNX model from `MODEL_URL`.
4. Prefer the WebGPU execution provider and fall back when needed.
5. Parse the corpus from the textarea.
6. Embed each sentence with explicit tokenization, inference, mean pooling, and normalization.
7. Compress the embedding matrix with `build_index(...)`.
8. Save `{ blob, sentences }` to IndexedDB.
9. At query time, embed the query and search the compressed index.

## Where to edit first

- `MODEL_URL`
- `TOKENIZER_URL`
- `DIM`
- `MAX_SEQ_LEN`
- `IDB_NAME`, `IDB_STORE`, `IDB_KEY`
- the corpus textarea content

If you swap the model, the tokenizer and output dimension must stay aligned with that new model.

## Why this is a better template than the Transformers.js demo for some teams

- You can pin and self-host model assets.
- You can replace the tokenizer logic.
- You can control ONNX Runtime settings directly.
- You can inspect tokenization, inference, and search latency separately.

That makes it a better starting point if you plan to own the full inference stack.

## Caveats

- The tokenizer is intentionally minimal. It is a readable example, not a hardened tokenizer package.
- Mean pooling and normalization are implemented in userland for clarity.
- The demo still embeds the full corpus in the page. For larger or more static corpora, precompute embeddings or even the compressed index offline.

## Good production follow-ups

- self-host the ONNX model and tokenizer assets instead of fetching from Hugging Face at runtime
- move embedding work into a worker if UI responsiveness matters
- persist metadata separately from the compressed index blob
- keep model version, tokenizer version, and index version tied together

## When to choose this example

Choose this example when you need a serious browser-side inference template.

If you only want the quickest route to a semantic-search proof of concept, [transformers-js](../transformers-js/README.md) is the better place to start.
