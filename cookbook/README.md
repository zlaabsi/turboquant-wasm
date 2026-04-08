# Cookbook

These guides are for turning TurboQuant into an application pattern, not just running a demo page.

## Choose a recipe

| If your problem looks like this | Start here |
|---|---|
| I already have embeddings and want static-site or in-browser search | [Browser Search](browser-search.md) |
| I want local retrieval over PDFs, notes, or imported files | [Client-Side RAG](client-rag.md) |
| I want an HTTP search API in a Worker or other edge runtime | [Edge and Serverless](edge-serverless.md) |
| I want to index tabs, bookmarks, or page text inside a browser extension | [Browser Extension](browser-extension.md) |
| I want Electron, Tauri, or mobile packaging patterns | [Desktop and Mobile](desktop-mobile.md) |

## Core building blocks

Most integrations reduce to the same primitives:

1. Pick an embedding dimension and quantization bit-width.
2. Produce a `Float32Array` of embeddings.
3. Build an index with `quantizer.buildIndex(...)` or the streaming API.
4. Store the compressed index separately from your metadata.
5. Reload with `Index.load(...)` and search by query embedding.

## What TurboQuant does not do for you

TurboQuant is the compression and retrieval layer. It does not decide:

- how you chunk documents
- how you compute embeddings
- how you version your corpus
- how you hydrate ids back into app data
- how you rerank final results

That separation is deliberate. The guides below are about stitching those pieces together cleanly.
