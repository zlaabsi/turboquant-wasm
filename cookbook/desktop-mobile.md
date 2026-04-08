# Desktop and Mobile

This guide covers the packaging patterns where TurboQuant is useful even if the embedding pipeline is different across platforms.

## Desktop: Electron and Tauri

Good fits:

- local file search
- note apps
- IDE or code search over precomputed embeddings

Recommended pattern:

1. Crawl or import files.
2. Chunk and embed them in a worker, helper process, or background task.
3. Build a TurboQuant index and save it to disk.
4. Load the saved index in the app shell for fast local search.
5. Rebuild or append only the segments that changed.

Important separation:

- TurboQuant index: compact retrieval layer
- app database: file paths, titles, previews, timestamps, permissions, UI state

Do not blur those together.

## Mobile

There are two different mobile stories:

### 1. On-device retrieval

Good when:

- the user already has embeddings on the device
- the runtime supports the WASM path you want
- privacy matters more than absolute throughput

### 2. Server-to-device compressed delivery

Good when:

- server-side embedding is acceptable
- mobile inference is too expensive or too slow
- bandwidth and storage matter

This is often the cleaner starting point.

## Transport compression pattern

Instead of sending raw float32 embeddings to the client:

1. generate embeddings on the server
2. compress them with TurboQuant
3. send the compressed bytes plus metadata
4. load them locally and search on-device

That can cut transfer size substantially. For example, a raw `384d` float32 vector is `1536 B`, while a `4-bit` TurboQuant representation is roughly `196 B` in the current implementation.

## React Native / Hermes caution

Treat React Native and Hermes as "validate runtime first", not "copy browser code and hope".

Check early:

- WASM support in the exact runtime you ship
- binary asset loading
- persistence APIs
- memory limits on the devices you care about

If any of those are awkward, the fallback architecture is usually:

- embeddings upstream
- compressed index delivered to device
- retrieval on device

## Practical desktop/mobile guidance

- Keep metadata outside the vector index.
- Version saved indexes by model and corpus revision.
- Prefer prebuilt indexes for large mostly-static corpora.
- Use streaming insertion only when user content changes incrementally.
- Benchmark on the actual target devices before you promise latency.

## Related use case

If your "desktop" target is actually a browser extension, use [Browser Extension](browser-extension.md) instead.
