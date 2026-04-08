# Desktop and Mobile

## Desktop use cases

- Electron or Tauri local file search.
- IDE symbol or code search over precomputed code embeddings.
- Local-first apps that need a compact retrieval layer.

## Mobile use cases

- Offline semantic search over user content.
- Shipping compressed vectors from server to device to cut bandwidth.

## Recommended architecture

1. Keep embedding generation and retrieval separate.
2. If mobile inference is expensive, generate embeddings upstream and only ship compressed vectors.
3. Use TurboQuant locally for retrieval and persistence.

## Notes

- Electron and Tauri are straightforward because they can host a browser-like or Node-like runtime.
- Mobile WASM support depends on the runtime you choose. Treat TurboQuant as the retrieval layer and validate runtime constraints early.
- The transport win is often meaningful: compressed vectors can be much smaller than raw `float32` embeddings, especially for 384d and 768d models.
