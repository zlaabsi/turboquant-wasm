/**
 * TurboQuant Cloudflare Worker — vector search at the edge.
 *
 * Builds a bag-of-words index from ~50 sentences on first request,
 * then serves search queries via GET /search?q=...
 *
 * Build: wasm-pack build --target web --out-dir pkg  (from turboquant-wasm root)
 * Dev:   wrangler dev
 */

import init, { TurboQuantizer, CompressedIndex } from '../../pkg/turboquant_wasm.js';

// ---------------------------------------------------------------------------
// Fixed vocabulary (~200 common English words)
// ---------------------------------------------------------------------------

const VOCABULARY = [
  'a', 'about', 'accuracy', 'across', 'after', 'algorithm', 'algorithms',
  'all', 'an', 'and', 'angle', 'angles', 'ann', 'approximate', 'are',
  'artificial', 'at', 'ball', 'based', 'behind', 'between', 'beyond',
  'billions', 'birds', 'brown', 'by', 'can', 'capture', 'cat', 'chased',
  'city', 'classification', 'climate', 'cloud', 'clustering', 'code', 'codes',
  'compact', 'compress', 'compressed', 'compression', 'compute', 'computing',
  'cosine', 'data', 'database', 'deep', 'dimensional', 'dimensionality',
  'dimensions', 'discover', 'discovery', 'distances', 'distortion',
  'distributed', 'documents', 'dog', 'drove', 'earth', 'edge', 'effective',
  'efficient', 'efficiently', 'embedding', 'embeddings', 'encode', 'engines',
  'every', 'exact', 'explores', 'exploring', 'fast', 'fell', 'fields', 'find',
  'finding', 'finds', 'flew', 'for', 'fox', 'from', 'generation', 'graphs',
  'green', 'handle', 'handling', 'has', 'hash', 'hashing', 'high', 'hills',
  'how', 'image', 'in', 'index', 'information', 'inner', 'intelligence',
  'into', 'is', 'items', 'its', 'jumps', 'key', 'knowledge', 'language',
  'large', 'lazy', 'learn', 'learning', 'looked', 'loss', 'lower', 'machine',
  'mapping', 'mat', 'matrices', 'matrix', 'meaning', 'measures', 'memory',
  'migration', 'minimal', 'minimize', 'model', 'models', 'modern', 'mountains',
  'natural', 'nearest', 'needs', 'neighbor', 'neighbors', 'networks', 'neural',
  'new', 'no', 'not', 'now', 'of', 'on', 'one', 'optimal', 'orange',
  'orthogonal', 'out', 'over', 'pages', 'painting', 'park', 'perfectly',
  'possible', 'power', 'preserve', 'preserves', 'processing', 'products',
  'projects', 'quality', 'quantization', 'query', 'quick', 'quiet', 'rain',
  'random', 'ranks', 'real', 'recognition', 'recommender', 'reduce', 'reduces',
  'reduction', 'relevance', 'representations', 'requires', 'research',
  'retrieval', 'revolutionized', 'rotation', 'sat', 'satellite', 'scales',
  'search', 'semantic', 'server', 'serverless', 'set', 'similar', 'similarity',
  'sky', 'smaller', 'softly', 'south', 'speed', 'storage', 'streets',
  'structure', 'sun', 'systems', 'technique', 'techniques', 'text', 'the',
  'through', 'time', 'to', 'trade', 'transformers', 'trees', 'two', 'usage',
  'use', 'using', 'vector', 'vectors', 'village', 'web', 'window', 'winter',
  'with', 'world',
];

const VOCAB_INDEX = new Map(VOCABULARY.map((w, i) => [w, i]));
const DIM = VOCABULARY.length;

// ---------------------------------------------------------------------------
// Pre-built corpus
// ---------------------------------------------------------------------------

const CORPUS = [
  'The quick brown fox jumps over the lazy dog',
  'Machine learning models compress high dimensional data',
  'Vector search finds similar items in a database',
  'Neural networks learn distributed representations',
  'The cat sat on the mat and looked out the window',
  'Approximate nearest neighbor search scales to billions',
  'Quantization reduces memory usage with minimal quality loss',
  'Embedding vectors capture semantic meaning of text',
  'The dog chased the ball across the green park',
  'Information retrieval ranks documents by relevance',
  'Transformers revolutionized natural language processing',
  'Cosine similarity measures the angle between two vectors',
  'The birds flew south for the winter migration',
  'Random rotation preserves inner products between vectors',
  'Compression algorithms trade accuracy for smaller storage',
  'The sun set behind the mountains painting the sky orange',
  'Search engines index billions of web pages for fast retrieval',
  'Orthogonal matrices preserve distances and angles perfectly',
  'The rain fell softly on the quiet village streets',
  'Dimensionality reduction projects data to lower dimensions',
  'Deep learning has revolutionized image recognition',
  'Hash based approximate nearest neighbor search is fast',
  'Knowledge graphs encode real world information',
  'Cloud computing scales machine learning to large data',
  'Serverless edge computing is the new computing model',
  'The earth satellite drove across the green hills',
  'Compact embedding models are effective for search',
  'Neural machine learning requires large data and compute',
  'Recommender systems find items similar to query',
  'Vector database systems handle high dimensional data',
  'Text embedding models capture semantic meaning',
  'Modern search engines use vector similarity',
  'Approximate search techniques trade accuracy for speed',
  'Random hashing is one key technique for fast search',
  'Clustering algorithms discover structure in data',
  'Compressed representations reduce memory and storage',
  'The city streets are quiet after the rain',
  'Language models learn from large text data',
  'Image classification using deep neural networks',
  'Semantic search finds documents by meaning not exact text',
  'Every vector database needs an efficient index',
  'Data compression is key to handling large embeddings',
  'Artificial intelligence research explores new models',
  'Real time vector search at the edge is now possible',
  'Embedding vectors into lower dimensions preserves similarity',
  'Finding nearest neighbors in high dimensional data',
  'The power of machine learning for information retrieval',
  'Optimal quantization algorithms minimize distortion',
  'Mapping text to vectors for semantic similarity search',
  'Generation of compact codes for vector search',
];

// ---------------------------------------------------------------------------
// Bag-of-words embedding
// ---------------------------------------------------------------------------

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(w => w.length > 0);
}

function textToVector(text) {
  const vec = new Float32Array(DIM);
  for (const word of tokenize(text)) {
    const idx = VOCAB_INDEX.get(word);
    if (idx !== undefined) vec[idx] += 1.0;
  }
  let norm = 0;
  for (let i = 0; i < DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < DIM; i++) vec[i] /= norm;
  }
  return vec;
}

// ---------------------------------------------------------------------------
// Index state (built once, reused across requests)
// ---------------------------------------------------------------------------

let quantizer = null;
let index = null;
let initialized = false;
let initError = null;

async function ensureIndex() {
  if (initialized) return;
  if (initError) throw initError;

  try {
    await init();

    quantizer = new TurboQuantizer(DIM, 4, BigInt(42));
    index = CompressedIndex.new_empty(DIM, 4);

    for (const sentence of CORPUS) {
      index.add_vector(quantizer, textToVector(sentence));
    }

    initialized = true;
  } catch (e) {
    initError = e;
    throw e;
  }
}

// ---------------------------------------------------------------------------
// HTML landing page
// ---------------------------------------------------------------------------

const HTML_PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TurboQuant Edge Search</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { font-size: 1.3rem; margin-bottom: 0.25rem; }
    .sub { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
    form { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    input { flex: 1; padding: 0.5rem 0.75rem; font-size: 0.95rem; border: 1px solid #ccc; border-radius: 6px; }
    input:focus { outline: 2px solid #2563eb; border-color: transparent; }
    button { background: #2563eb; color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.95rem; }
    button:hover { background: #1d4ed8; }
    #results { margin-top: 0.5rem; }
    .item { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #eee; }
    .rank { color: #999; font-size: 0.8rem; min-width: 1.5rem; }
    .text { flex: 1; margin: 0 0.5rem; }
    .score { font-family: monospace; font-size: 0.85rem; color: #2563eb; white-space: nowrap; }
    .meta { margin-top: 1rem; font-size: 0.8rem; color: #888; font-family: monospace; }
    .note { margin-top: 1.5rem; padding: 0.75rem; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 6px; font-size: 0.8rem; color: #0369a1; }
  </style>
</head>
<body>
  <h1>TurboQuant Edge Search</h1>
  <p class="sub">Vector search running in a Cloudflare Worker via WASM &mdash; ${CORPUS.length} sentences, dim=${DIM}, 4-bit quantization</p>
  <form id="f" onsubmit="return doSearch(event)">
    <input type="text" id="q" name="q" placeholder="Type a search query..." value="finding similar vectors quickly" autofocus>
    <button type="submit">Search</button>
  </form>
  <div id="results"></div>
  <div id="meta" class="meta"></div>
  <div class="note">
    This demo uses bag-of-words embeddings (fixed ${DIM}-word vocabulary).
    Real applications use neural embedding models for better semantic matching.
    The point is to show TurboQuant WASM running at the edge with zero cold-start overhead.
  </div>
  <script>
    async function doSearch(e) {
      e.preventDefault();
      const q = document.getElementById('q').value.trim();
      if (!q) return false;
      const res = await fetch('/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      if (data.error) {
        document.getElementById('results').innerHTML = '<p style="color:red">' + data.error + '</p>';
        return false;
      }
      let html = '';
      for (const r of data.results) {
        html += '<div class="item"><span class="rank">' + r.rank + '.</span><span class="text">' + r.text + '</span><span class="score">' + r.score.toFixed(4) + '</span></div>';
      }
      document.getElementById('results').innerHTML = html;
      document.getElementById('meta').textContent = 'Search took ' + data.search_ms.toFixed(2) + ' ms | ' + data.n_vectors + ' vectors | ' + data.memory_bytes + ' bytes';
      return false;
    }
  </script>
</body>
</html>`;

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request) {
    const url = new URL(request.url);

    try {
      await ensureIndex();
    } catch (e) {
      return Response.json({ error: 'Failed to initialize index: ' + e.message }, { status: 500 });
    }

    if (url.pathname === '/search') {
      return handleSearch(url);
    }

    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(HTML_PAGE, {
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      });
    }

    return Response.json({ error: 'Not found. Try GET / or GET /search?q=...' }, { status: 404 });
  },
};

function handleSearch(url) {
  const q = url.searchParams.get('q');
  if (!q || !q.trim()) {
    return Response.json({ error: 'Missing query parameter: ?q=...' }, { status: 400 });
  }

  const queryVec = textToVector(q);

  // Check vocabulary overlap
  const queryWords = tokenize(q);
  const matched = queryWords.filter(w => VOCAB_INDEX.has(w));
  if (matched.length === 0) {
    return Response.json({
      error: 'No vocabulary overlap. Try common English words.',
      query: q,
      results: [],
    });
  }

  const k = Math.min(5, CORPUS.length);
  const t0 = performance.now();
  const topK = index.search(quantizer, queryVec, k);
  const searchMs = performance.now() - t0;

  const results = [];
  for (let rank = 0; rank < topK.length; rank++) {
    const idx = topK[rank];
    const sentVec = textToVector(CORPUS[idx]);
    let score = 0;
    for (let j = 0; j < DIM; j++) score += sentVec[j] * queryVec[j];

    results.push({
      rank: rank + 1,
      index: idx,
      text: CORPUS[idx],
      score,
      matched_words: tokenize(CORPUS[idx]).filter(w => matched.includes(w)),
    });
  }

  return Response.json({
    query: q,
    results,
    search_ms: searchMs,
    n_vectors: index.n_vectors,
    memory_bytes: index.memory_bytes,
    vocabulary_size: DIM,
    bits: 4,
  });
}
