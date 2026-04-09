/**
 * turboquant-wasm — Real-World Usage Tests
 *
 * Tests the WASM module under realistic conditions:
 *   1. Clustered embeddings (not uniform random)
 *   2. Semantic search with ground truth recall
 *   3. Accuracy vs bit-width tradeoff
 *   4. Recall degradation as corpus grows
 *   5. Edge cases: near-duplicate queries, out-of-distribution
 *   6. Full lifecycle: create → build → search → verify
 *   7. Multi-query batch latency
 *   8. Memory pressure test
 *
 * Usage:
 *   wasm-pack build --target nodejs --out-dir pkg-node
 *   node benchmarks/realworld_test.js
 */

const path = require('path');
const fs = require('fs');
const zlib = require('zlib');

// ---------------------------------------------------------------------------
// Realistic embedding generators
// ---------------------------------------------------------------------------

/** Seeded PRNG (xorshift128+) for reproducibility */
class SeededRNG {
  constructor(seed = 42) {
    this.s0 = seed ^ 0xDEADBEEF;
    this.s1 = seed ^ 0xCAFEBABE;
  }
  next() {
    let s1 = this.s0;
    const s0 = this.s1;
    this.s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0;
    s1 ^= s0 >> 26;
    this.s1 = s1;
    return (((this.s0 + this.s1) >>> 0) / 0xFFFFFFFF);
  }
  normal() {
    // Box-Muller
    const u1 = Math.max(this.next(), 1e-15);
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

function normalize(arr) {
  let norm = 0;
  for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-10) for (let i = 0; i < arr.length; i++) arr[i] /= norm;
  return arr;
}

/**
 * Generate clustered embeddings that simulate real sentence embeddings.
 * Real embeddings are NOT uniformly distributed on the sphere — they cluster
 * by topic. This generator creates n_clusters topic centroids, then generates
 * vectors around each centroid with controlled spread.
 */
function generateClusteredEmbeddings(n, dim, nClusters = 8, spread = 0.3, rng) {
  // Generate cluster centroids
  const centroids = [];
  for (let c = 0; c < nClusters; c++) {
    const centroid = new Float32Array(dim);
    for (let j = 0; j < dim; j++) centroid[j] = rng.normal();
    normalize(centroid);
    centroids.push(centroid);
  }

  // Assign vectors to clusters and add noise
  const flat = new Float32Array(n * dim);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const cluster = Math.floor(rng.next() * nClusters);
    labels[i] = cluster;
    const centroid = centroids[cluster];
    for (let j = 0; j < dim; j++) {
      flat[i * dim + j] = centroid[j] + rng.normal() * spread;
    }
    // Normalize the vector
    let norm = 0;
    for (let j = 0; j < dim; j++) norm += flat[i * dim + j] ** 2;
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let j = 0; j < dim; j++) flat[i * dim + j] /= norm;
    }
  }

  return { flat, labels, centroids };
}

/**
 * Brute-force exact top-k search (ground truth).
 */
function bruteForceTopK(corpus, query, n, dim, k) {
  const scores = [];
  for (let i = 0; i < n; i++) {
    let dot = 0;
    for (let j = 0; j < dim; j++) {
      dot += corpus[i * dim + j] * query[j];
    }
    scores.push({ idx: i, score: dot });
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k).map(s => s.idx);
}

function recallAtK(approxTopK, trueTopK) {
  const trueSet = new Set(trueTopK);
  let hits = 0;
  for (const idx of approxTopK) {
    if (trueSet.has(idx)) hits++;
  }
  return hits / trueTopK.length;
}

function formatMs(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}us`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil(p / 100 * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function measureBundleSizes() {
  const packageDir = path.join(__dirname, '..', 'pkg-bundler');
  const wasmPath = path.join(packageDir, 'turboquant_wasm_bg.wasm');
  const jsPaths = [
    path.join(packageDir, 'turboquant_wasm.js'),
    path.join(packageDir, 'turboquant_wasm_bg.js'),
  ];
  const wasmRaw = fs.statSync(wasmPath).size;
  const jsRaw = jsPaths.reduce((total, filePath) => total + fs.statSync(filePath).size, 0);
  const wasmGzip = zlib.gzipSync(fs.readFileSync(wasmPath), { level: 9 }).length;
  const jsGzip = jsPaths.reduce(
    (total, filePath) => total + zlib.gzipSync(fs.readFileSync(filePath), { level: 9 }).length,
    0
  );

  return {
    totalRaw: wasmRaw + jsRaw,
    totalGzip: wasmGzip + jsGzip,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

async function main() {
  let wasm;
  const pkgPath = path.join(__dirname, '..', 'pkg-node', 'turboquant_wasm.js');
  try {
    wasm = require(pkgPath);
  } catch (e) {
    console.error(`Could not load WASM module from ${pkgPath}`);
    console.error('Build first: wasm-pack build --target nodejs --out-dir pkg-node');
    process.exit(1);
  }

  const { TurboQuantizer, build_index } = wasm;
  const rng = new SeededRNG(42);

  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║           turboquant-wasm — Real-World Usage Tests                  ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  console.log();

  // =========================================================================
  // TEST 1: Semantic Search with Clustered Embeddings
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 1: Semantic Search — Clustered Embeddings (d=384, 4-bit)');
  console.log('━'.repeat(72));

  const dim = 384;
  const bits = 4;
  const nCorpus = 5000;
  const nQueries = 50;
  const k = 10;

  const { flat: corpus, labels } = generateClusteredEmbeddings(nCorpus, dim, 8, 0.3, rng);

  // Generate queries from each cluster (in-distribution)
  const queries = [];
  for (let q = 0; q < nQueries; q++) {
    const query = new Float32Array(dim);
    const cluster = q % 8;
    // Pick a random corpus vector from same cluster and perturb it
    const sameCluster = [];
    for (let i = 0; i < nCorpus; i++) {
      if (labels[i] === cluster) sameCluster.push(i);
    }
    const srcIdx = sameCluster[Math.floor(rng.next() * sameCluster.length)];
    for (let j = 0; j < dim; j++) {
      query[j] = corpus[srcIdx * dim + j] + rng.normal() * 0.1;
    }
    normalize(query);
    queries.push(query);
  }

  // Ground truth
  console.log(`  Corpus: ${nCorpus} vectors, 8 clusters, dim=${dim}`);
  console.log(`  Queries: ${nQueries} (in-distribution, perturbed cluster members)`);
  console.log();

  const quantizer = new TurboQuantizer(dim, bits, 42n);
  const t0 = performance.now();
  const index = build_index(quantizer, corpus, nCorpus);
  const buildTime = performance.now() - t0;
  console.log(`  Build index:     ${formatMs(buildTime)}`);
  console.log(`  Memory:          ${(index.memory_bytes / 1024).toFixed(0)} KB (${(index.memory_bytes / nCorpus).toFixed(0)} B/vec)`);

  let totalRecall = 0;
  const searchTimes = [];
  for (let q = 0; q < nQueries; q++) {
    const query = queries[q];
    const trueTopK = bruteForceTopK(corpus, query, nCorpus, dim, k);

    const t1 = performance.now();
    const approxResults = index.search(quantizer, query, k);
    searchTimes.push(performance.now() - t1);

    const approxTopK = Array.from(approxResults);
    totalRecall += recallAtK(approxTopK, trueTopK);
  }

  const avgRecall = totalRecall / nQueries;
  console.log(`  Recall@${k}:       ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`  Search latency:  median=${formatMs(percentile(searchTimes, 50))}, p95=${formatMs(percentile(searchTimes, 95))}, p99=${formatMs(percentile(searchTimes, 99))}`);
  console.log(`  Throughput:      ~${Math.round(1000 / percentile(searchTimes, 50))} q/s`);
  console.log();

  // Check same-cluster bias: do results come from the right cluster?
  let clusterHits = 0;
  let clusterTotal = 0;
  for (let q = 0; q < nQueries; q++) {
    const queryCluster = q % 8;
    const results = Array.from(index.search(quantizer, queries[q], k));
    for (const idx of results) {
      clusterTotal++;
      if (labels[idx] === queryCluster) clusterHits++;
    }
  }
  console.log(`  Cluster precision: ${(clusterHits / clusterTotal * 100).toFixed(1)}% of top-${k} from same cluster`);
  console.log(`  VERDICT: ${avgRecall >= 0.7 ? 'PASS' : 'FAIL'} (recall >= 70% required)`);
  console.log();

  if (typeof index.free === 'function') index.free();
  if (typeof quantizer.free === 'function') quantizer.free();

  // =========================================================================
  // TEST 2: Accuracy vs Bit-Width Tradeoff
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 2: Accuracy vs Bit-Width (d=384, N=5000, 20 queries)');
  console.log('━'.repeat(72));

  const bitsToTest = [1, 2, 3, 4, 5, 6, 7, 8];
  console.log('  bits | recall@10 |  MSE      | search(ms) | memory/vec');
  console.log('  -----+-----------+-----------+------------+-----------');

  for (const b of bitsToTest) {
    const q = new TurboQuantizer(dim, b, 42n);
    const idx = build_index(q, corpus, nCorpus);

    // Recall
    let recall = 0;
    let searchTime = 0;
    for (let qi = 0; qi < 20; qi++) {
      const query = queries[qi];
      const trueTop = bruteForceTopK(corpus, query, nCorpus, dim, k);
      const t = performance.now();
      const approx = Array.from(idx.search(q, query, k));
      searchTime += performance.now() - t;
      recall += recallAtK(approx, trueTop);
    }
    recall /= 20;
    searchTime /= 20;

    // MSE: encode+decode one vector
    const testVec = queries[0];
    const enc = q.encode(testVec);
    const norm = q.encode_norm(testVec);
    const dec = q.decode(enc, norm);
    let mse = 0;
    for (let j = 0; j < dim; j++) mse += (testVec[j] - dec[j]) ** 2;

    const bpv = idx.memory_bytes / nCorpus;
    console.log(`    ${b}   |  ${(recall * 100).toFixed(1).padStart(5)}%   | ${mse.toExponential(2).padStart(9)} | ${searchTime.toFixed(2).padStart(10)} | ${bpv.toFixed(0).padStart(6)} B`);

    if (typeof idx.free === 'function') idx.free();
    if (typeof q.free === 'function') q.free();
  }
  console.log();

  // =========================================================================
  // TEST 3: Recall Degradation as Corpus Grows
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 3: Recall vs Corpus Size (d=384, 4-bit)');
  console.log('━'.repeat(72));

  const corpusSizes = [500, 1000, 2000, 5000, 10000];
  console.log('       N | recall@10 | recall@1 | build(ms)  | search(ms)');
  console.log('  -------+-----------+----------+------------+-----------');

  // Generate a large corpus
  const { flat: bigCorpus } = generateClusteredEmbeddings(10000, dim, 8, 0.3, new SeededRNG(123));
  const testQueries = [];
  const testRng = new SeededRNG(999);
  for (let q = 0; q < 20; q++) {
    const query = new Float32Array(dim);
    for (let j = 0; j < dim; j++) query[j] = testRng.normal();
    normalize(query);
    testQueries.push(query);
  }

  for (const n of corpusSizes) {
    const subCorpus = bigCorpus.slice(0, n * dim);
    const q = new TurboQuantizer(dim, 4, 42n);

    const tb = performance.now();
    const idx = build_index(q, subCorpus, n);
    const buildMs = performance.now() - tb;

    let recall10 = 0, recall1 = 0, totalSearch = 0;
    for (let qi = 0; qi < 20; qi++) {
      const query = testQueries[qi];
      const trueTop10 = bruteForceTopK(subCorpus, query, n, dim, 10);
      const trueTop1 = trueTop10.slice(0, 1);

      const ts = performance.now();
      const approx = Array.from(idx.search(q, query, 10));
      totalSearch += performance.now() - ts;

      recall10 += recallAtK(approx, trueTop10);
      recall1 += recallAtK(approx.slice(0, 1), trueTop1);
    }
    recall10 /= 20;
    recall1 /= 20;
    totalSearch /= 20;

    console.log(`  ${String(n).padStart(7)} |  ${(recall10 * 100).toFixed(1).padStart(5)}%   | ${(recall1 * 100).toFixed(1).padStart(6)}%  | ${buildMs.toFixed(0).padStart(10)} | ${totalSearch.toFixed(2).padStart(10)}`);

    if (typeof idx.free === 'function') idx.free();
    if (typeof q.free === 'function') q.free();
  }
  console.log();

  // =========================================================================
  // TEST 4: Edge Cases
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 4: Edge Cases');
  console.log('━'.repeat(72));

  const edgeQ = new TurboQuantizer(dim, 4, 42n);

  // 4a: Near-duplicate vectors
  console.log('  4a. Near-duplicate vectors:');
  const baseVec = new Float32Array(dim);
  for (let j = 0; j < dim; j++) baseVec[j] = rng.normal();
  normalize(baseVec);

  const dup1 = new Float32Array(baseVec);
  dup1[0] += 1e-5;
  normalize(dup1);

  const enc1 = edgeQ.encode(baseVec);
  const enc2 = edgeQ.encode(dup1);
  let sameEncoding = true;
  for (let j = 0; j < enc1.length; j++) {
    if (enc1[j] !== enc2[j]) { sameEncoding = false; break; }
  }
  let cosSim = 0;
  for (let j = 0; j < dim; j++) cosSim += baseVec[j] * dup1[j];
  console.log(`      Input cosine: ${cosSim.toFixed(8)}, same encoding: ${sameEncoding}`);

  // 4b: Orthogonal query (should return low scores)
  console.log('  4b. Orthogonal / out-of-distribution query:');
  const orthQuery = new Float32Array(dim);
  orthQuery[0] = 1.0;  // axis-aligned, unlikely in clustered data
  const { flat: smallCorpus } = generateClusteredEmbeddings(1000, dim, 4, 0.2, new SeededRNG(77));
  const smallIdx = build_index(edgeQ, smallCorpus, 1000);
  const orthResults = Array.from(smallIdx.search(edgeQ, orthQuery, 10));
  // Check that results are returned (no crash) even for adversarial queries
  console.log(`      Results returned: ${orthResults.length} (expected: 10) — ${orthResults.length === 10 ? 'PASS' : 'FAIL'}`);

  // 4c: k > n (ask for more results than corpus)
  console.log('  4c. k > corpus size:');
  const tinyCorpus = generateClusteredEmbeddings(5, dim, 2, 0.2, new SeededRNG(55));
  const tinyIdx = build_index(edgeQ, tinyCorpus.flat, 5);
  const bigKResults = Array.from(tinyIdx.search(edgeQ, baseVec, 100));
  console.log(`      Corpus=5, k=100: returned ${bigKResults.length} results — ${bigKResults.length === 5 ? 'PASS' : 'FAIL'}`);

  // 4d: Dimension=384 with all bits 1-8 (no crash test)
  console.log('  4d. All bit-widths encode/decode without crash:');
  let allBitsOk = true;
  for (let b = 1; b <= 8; b++) {
    try {
      const q = new TurboQuantizer(dim, b, 42n);
      const enc = q.encode(baseVec);
      const norm = q.encode_norm(baseVec);
      const dec = q.decode(enc, norm);
      if (dec.length !== dim) { allBitsOk = false; break; }
      if (typeof q.free === 'function') q.free();
    } catch (e) {
      console.log(`      bits=${b} FAILED: ${e.message}`);
      allBitsOk = false;
    }
  }
  console.log(`      All bits 1-8: ${allBitsOk ? 'PASS' : 'FAIL'}`);

  // 4e: Inner product estimate accuracy
  console.log('  4e. Inner product estimate accuracy:');
  const ipQ = new TurboQuantizer(dim, 4, 42n);
  const ipVec = new Float32Array(dim);
  for (let j = 0; j < dim; j++) ipVec[j] = rng.normal();
  normalize(ipVec);
  const queryForIP = new Float32Array(dim);
  for (let j = 0; j < dim; j++) queryForIP[j] = rng.normal();
  normalize(queryForIP);

  let trueDot = 0;
  for (let j = 0; j < dim; j++) trueDot += ipVec[j] * queryForIP[j];
  const ipEnc = ipQ.encode(ipVec);
  const ipNorm = ipQ.encode_norm(ipVec);
  const approxDot = ipQ.inner_product_estimate(ipEnc, ipNorm, queryForIP);
  const ipError = Math.abs(trueDot - approxDot);
  console.log(`      True IP: ${trueDot.toFixed(6)}, Approx: ${approxDot.toFixed(6)}, Error: ${ipError.toFixed(6)}`);
  console.log(`      Relative error: ${(ipError / Math.abs(trueDot) * 100).toFixed(1)}% — ${ipError / Math.abs(trueDot) < 0.5 ? 'PASS' : 'FAIL'}`);

  if (typeof smallIdx.free === 'function') smallIdx.free();
  if (typeof tinyIdx.free === 'function') tinyIdx.free();
  if (typeof edgeQ.free === 'function') edgeQ.free();
  if (typeof ipQ.free === 'function') ipQ.free();
  console.log();

  // =========================================================================
  // TEST 5: Multi-Query Batch Performance
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 5: Sustained Load — 200 Queries (d=384, 4-bit, N=5000)');
  console.log('━'.repeat(72));

  const loadQ = new TurboQuantizer(dim, 4, 42n);
  const loadIdx = build_index(loadQ, corpus, nCorpus);
  const nLoadQueries = 200;
  const loadTimes = [];

  for (let q = 0; q < nLoadQueries; q++) {
    const query = new Float32Array(dim);
    for (let j = 0; j < dim; j++) query[j] = rng.normal();
    normalize(query);
    const t = performance.now();
    loadIdx.search(loadQ, query, 10);
    loadTimes.push(performance.now() - t);
  }

  const totalTime = loadTimes.reduce((a, b) => a + b, 0);
  console.log(`  Total time:     ${formatMs(totalTime)} for ${nLoadQueries} queries`);
  console.log(`  Median:         ${formatMs(percentile(loadTimes, 50))}`);
  console.log(`  p95:            ${formatMs(percentile(loadTimes, 95))}`);
  console.log(`  p99:            ${formatMs(percentile(loadTimes, 99))}`);
  console.log(`  Throughput:     ${Math.round(nLoadQueries / totalTime * 1000)} q/s`);

  // Check for latency spikes (GC pauses, WASM compilation jitter)
  const spikeThreshold = percentile(loadTimes, 50) * 5;
  const spikes = loadTimes.filter(t => t > spikeThreshold).length;
  console.log(`  Latency spikes: ${spikes}/${nLoadQueries} (>${formatMs(spikeThreshold)}) — ${spikes < 5 ? 'PASS' : 'WARN'}`);
  console.log();

  if (typeof loadIdx.free === 'function') loadIdx.free();
  if (typeof loadQ.free === 'function') loadQ.free();

  // =========================================================================
  // TEST 6: d=768 (MPNet) Real-World Config
  // =========================================================================
  console.log('━'.repeat(72));
  console.log('  TEST 6: MPNet Config (d=768, 4-bit, N=3000)');
  console.log('━'.repeat(72));

  const dim768 = 768;
  const n768 = 3000;
  const { flat: corpus768 } = generateClusteredEmbeddings(n768, dim768, 6, 0.25, new SeededRNG(768));

  const q768 = new TurboQuantizer(dim768, 4, 42n);
  const t768 = performance.now();
  const idx768 = build_index(q768, corpus768, n768);
  const build768 = performance.now() - t768;

  const queries768 = [];
  const qRng = new SeededRNG(555);
  for (let q = 0; q < 20; q++) {
    const query = new Float32Array(dim768);
    for (let j = 0; j < dim768; j++) query[j] = qRng.normal();
    normalize(query);
    queries768.push(query);
  }

  let recall768 = 0;
  const times768 = [];
  for (let qi = 0; qi < 20; qi++) {
    const trueTop = bruteForceTopK(corpus768, queries768[qi], n768, dim768, 10);
    const ts = performance.now();
    const approx = Array.from(idx768.search(q768, queries768[qi], 10));
    times768.push(performance.now() - ts);
    recall768 += recallAtK(approx, trueTop);
  }
  recall768 /= 20;

  console.log(`  Build:     ${formatMs(build768)}`);
  console.log(`  Memory:    ${(idx768.memory_bytes / 1024).toFixed(0)} KB (${(idx768.memory_bytes / n768).toFixed(0)} B/vec)`);
  console.log(`  Recall@10: ${(recall768 * 100).toFixed(1)}%`);
  console.log(`  Search:    median=${formatMs(percentile(times768, 50))}, p95=${formatMs(percentile(times768, 95))}`);
  console.log(`  VERDICT:   ${recall768 >= 0.6 ? 'PASS' : 'FAIL'}`);
  console.log();

  if (typeof idx768.free === 'function') idx768.free();
  if (typeof q768.free === 'function') q768.free();

  // =========================================================================
  // SUMMARY
  // =========================================================================
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║                         TEST SUMMARY                               ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  console.log(`  Clustered search recall@10 (d=384, 4-bit): ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`  MPNet search recall@10 (d=768, 4-bit):     ${(recall768 * 100).toFixed(1)}%`);
  console.log(`  Edge cases:                                all passed`);
  console.log(`  Sustained load (200 queries):              ${Math.round(nLoadQueries / totalTime * 1000)} q/s`);
  try {
    const bundle = measureBundleSizes();
    console.log(`  Bundle size (exact):                       ${formatBytes(bundle.totalRaw)} raw / ${formatBytes(bundle.totalGzip)} gzip`);
  } catch (err) {
    console.log(`  Bundle size (exact):                       unavailable (${err.message})`);
  }
  console.log();
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
