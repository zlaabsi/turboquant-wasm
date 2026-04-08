# Edge and Serverless

## Good fits

- Cloudflare Worker search API.
- Edge functions serving a small or medium prebuilt index.
- WASM-friendly local runtimes on small devices.

## Recommended architecture

1. Precompute embeddings offline.
2. Compress them ahead of time.
3. Load the compressed index into the runtime on cold start or first request.
4. Embed or map incoming queries, then search in-memory.

## Pattern

- Browser and edge: use the web target and `pkg/`.
- Node-like runtimes: use the node target and `pkg-node/`.
- For HTTP APIs, return ids and scores, then hydrate metadata outside the index.

## Notes

- The [Cloudflare Worker example](../examples/cloudflare/README.md) is the canonical deployment pattern in this repo.
- For edge runtimes with strict memory budgets, keep the metadata store separate from the compressed vector index.
- For IoT or embedded deployments, TurboQuant is best when embeddings are generated elsewhere and only search happens on-device.
