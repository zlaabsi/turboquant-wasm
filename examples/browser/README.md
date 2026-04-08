# Browser Example

This is the smallest possible end-to-end demo in the repo. It does **not** use a neural embedding model. Instead, it builds a bag-of-words vector for each sentence and then indexes those vectors with TurboQuant.

## What it is good for

- verifying that the WASM module loads correctly
- understanding the raw `TurboQuantizer` + `CompressedIndex` flow
- seeing the streaming insertion API with `new_empty()` and `add_vector()`
- debugging compression and search without model downloads or inference noise

## What it is not

- not a semantic search demo
- not representative of production retrieval quality
- not a good baseline for comparing model-level search relevance

If you want real semantic search, start from [transformers-js](../transformers-js/README.md) or [onnx-webgpu](../onnx-webgpu/README.md).

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/browser/`.

## Flow

1. Read the corpus from the textarea.
2. Build a vocabulary from the words that appear in that corpus.
3. Convert each sentence into a normalized bag-of-words vector.
4. Create `TurboQuantizer(dim, bits, seed)`.
5. Create `CompressedIndex.new_empty(dim, bits)`.
6. Insert one vector at a time with `index.add_vector(...)`.
7. Build a query vector from the search box and call `index.search(...)`.

That makes this example the best place to understand the retrieval API itself.

## Where to modify it

- Change the quantization bit-width via `const BITS = 4`.
- Replace the demo corpus by editing the textarea content used by `document.getElementById('corpus').value`.
- Change tokenization or weighting in `tokenize()`, `buildVocabulary()`, and `textToVector()`.

## Why the streaming API matters

This example uses the incremental path instead of `build_index(...)` on purpose. If your app ingests documents one by one, the same pattern can be reused with real embeddings:

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.createEmptyIndex();

for (const embedding of embeddings) {
  index.addVector(embedding);
}
```

## Limitations

- Query terms must overlap the corpus vocabulary.
- Relevance is lexical, not semantic.
- The example computes display scores from reconstructed bag-of-words behavior, not from a neural model.

## How to turn this into something real

- Keep the same search and persistence pattern.
- Replace `textToVector()` with embeddings produced elsewhere.
- Load a prebuilt compressed index on startup instead of rebuilding from the textarea.
- Store document metadata separately from the vector index.

For that path, see [Browser Search](../../cookbook/browser-search.md).
