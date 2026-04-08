# Cookbook

Use these guides as architecture templates around the core API.

## Guides

- [Browser Search](browser-search.md)
- [Client-Side RAG](client-rag.md)
- [Edge and Serverless](edge-serverless.md)
- [Desktop and Mobile](desktop-mobile.md)

## API building blocks

Most integrations use the same primitives:

1. Create a quantizer with the embedding dimension.
2. Build or stream a compressed index.
3. Persist the index with `save()` if needed.
4. Reload with `Index.load()` or the raw WASM `CompressedIndex.load()`.
5. Search with a query embedding and map result ids back to documents.
