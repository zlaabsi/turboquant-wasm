# Browser Example

Zero-dependency demo using bag-of-words embeddings and TurboQuant compression.

## What it demonstrates

- Build a `TurboQuantizer` directly in the browser.
- Stream vectors into `CompressedIndex`.
- Search entirely client-side.
- Validate the API without downloading an embedding model.

## Run

From the repo root:

```bash
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080/examples/browser/`.

## When to use this example

- Quick sanity checks.
- Embedding-free demos.
- Static-site prototypes where the embedding step happens offline.
