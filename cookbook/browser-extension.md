# Browser Extension

Use this pattern when you want to index content that already lives inside the user's browser: tabs, bookmarks, saved pages, snippets, or notes.

## Good fits

- semantic search over open tabs
- local bookmark search
- personal web-memory or clipping tools
- "search my current workspace" browser utilities

## Recommended architecture

```text
content scripts / tabs API -> text extraction -> embedder -> TurboQuant index in extension storage
                                                        |
popup / side panel query -------------------------------+-> top-k ids -> tab/bookmark metadata
```

## Where TurboQuant fits

TurboQuant should live in the background service worker or extension backend process, not in every content script.

That backend owns:

- the compressed index
- the metadata table
- persistence
- search RPC from popup or side panel

## Suggested storage split

- metadata: tab id, URL, title, snippet, timestamps
- compressed index bytes: one blob from `index.save()`
- manifest: model version, dimension, corpus revision

Use IndexedDB when possible. Fall back to extension storage only if your index is very small.

## Update pattern

For tab indexing, the streaming API is usually the right shape:

```ts
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = quantizer.createEmptyIndex();
const metadata = [];

for (const tabDocument of extractedTabs) {
  const embedding = await embed(tabDocument.text);
  index.addVector(embedding);
  metadata.push({ tabId: tabDocument.id, url: tabDocument.url, title: tabDocument.title });
}

await saveExtensionState({
  index: index.save(),
  metadata,
});
```

## Query pattern

```ts
const state = await loadExtensionState();
const quantizer = await createQuantizer({ dim: 384, bits: 4 });
const index = Index.load(state.index, quantizer);

const queryEmbedding = await embed(queryText);
const ids = index.search(queryEmbedding, 8);
const hits = ids.map((i) => state.metadata[i]);
```

## Extension-specific cautions

- MV3 service workers are ephemeral. Load or rebuild state explicitly.
- Do not put heavy inference in content scripts unless you really need page-local isolation.
- Watch storage quotas early.
- Keep extracted page text normalized and size-limited before embedding.

## Best starting point in this repo

- API shape: [examples/browser](../examples/browser/README.md)
- real local embeddings: [examples/transformers-js](../examples/transformers-js/README.md)
- browser deployment pattern: [Browser Search](browser-search.md)
