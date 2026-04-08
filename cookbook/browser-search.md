# Browser Search

## Good fits

- Static documentation search.
- Blog or ecommerce search on a static frontend.
- Offline PWA search.
- Browser extensions that keep the index on-device.

## Recommended architecture

1. Compute embeddings offline or at build time.
2. Ship the compressed index or the raw embeddings with the app.
3. Build `TurboQuantizer` on startup.
4. Search inside the browser tab with no backend round-trip.

## Pattern

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.buildIndex(embeddings, documents.length);
const hits = index.search(queryEmbedding, 8);
```

## Persistence strategy

- Small indexes: keep in memory or store the serialized bytes in IndexedDB.
- Medium indexes: download a versioned binary index, then load it on demand.

## Notes

- The [browser example](../examples/browser/README.md) is the simplest starting point.
- If you already have embeddings from a separate pipeline, TurboQuant can be the last-mile compression and retrieval layer.
