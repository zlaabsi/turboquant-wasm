# Client-Side RAG

Use this pattern when the user loads local content and you want chunk retrieval to stay on-device.

Typical fits:

- PDF chat with privacy constraints
- note-taking or knowledge-base apps
- local document assistants in browser or desktop-like shells

## Architecture

```text
files -> parser -> chunker -> embedder -> TurboQuant index -> IndexedDB/local file
                                               |
query -> embedder -----------------------------+-> top-k chunk ids -> prompt assembly
```

TurboQuant is the chunk-index layer. It is not the parser, the chunker, or the reranker.

## Recommended data model

Persist these separately:

- `chunks[]`: `{ id, docId, text, page, section, ... }`
- `index bytes`: `index.save()`
- `manifest`: embedding model, dimension, bits, chunking parameters, source revision

That lets you invalidate safely when you change chunking or models.

## Ingestion pattern

For user-imported content, the streaming API is often the cleanest:

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.createEmptyIndex();
const chunks = [];

for await (const chunk of chunkStream) {
  const embedding = await embed(chunk.text);
  index.addVector(embedding);
  chunks.push(chunk);
}

await saveToIndexedDb({
  manifest: { dim: 384, bits: 4, model: "all-MiniLM-L6-v2", chunkVersion: "v1" },
  chunks,
  index: index.save(),
});
```

## Query pattern

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const saved = await loadFromIndexedDb();

const index = Index.load(saved.index, quantizer);
const queryEmbedding = await embed(userQuestion);
const ids = index.search(queryEmbedding, 6);
const retrievedChunks = ids.map((i) => saved.chunks[i]);
```

## Practical decisions

- Start with `384d` and `4-bit` unless you have a reason not to.
- Keep chunk text outside the vector blob.
- Store chunk ids in a stable order so the search result ids stay meaningful.
- Rebuild the index when you change model, tokenizer, chunk size, or overlap.
- If imports are large, move parsing and embedding off the main UI thread.

## PDF-specific notes

- Chunk by semantic section or paragraph ranges, not raw fixed bytes.
- Keep page number and source offsets in metadata for citation UI.
- Persist the raw document separately from the retrieval index.

## What TurboQuant is responsible for

- compact storage of chunk embeddings
- fast local top-k retrieval over those embeddings
- serializing and restoring the compressed index

## What stays outside TurboQuant

- file parsing
- OCR
- chunking strategy
- embedding model choice
- prompt assembly and answer generation

## Best matching repo resources

- fastest browser prototype: [examples/transformers-js](../examples/transformers-js/README.md)
- lower-level inference control: [examples/onnx-webgpu](../examples/onnx-webgpu/README.md)

## Failure modes to plan for

- model mismatch between saved index and current query embedder
- corpus edits without index rebuild
- storing too much duplicated chunk metadata in IndexedDB
- treating retrieval quality problems as quantization problems when the real issue is chunking or embeddings
