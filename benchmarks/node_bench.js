/**
 * turboquant-wasm Node.js Benchmark
 *
 * Measures real performance of the WASM module for various
 * configurations. Requires a Node.js target build:
 *
 *   wasm-pack build --target nodejs --out-dir pkg-node
 *   node benchmarks/node_bench.js
 */

const path = require('path');
const fs = require('fs');
const zlib = require('zlib');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randomFloat32Array(len) {
  const arr = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    arr[i] = (Math.random() - 0.5) * 2; // uniform in [-1, 1]
  }
  return arr;
}

function normalizeInPlace(arr) {
  let norm = 0;
  for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-10) {
    for (let i = 0; i < arr.length; i++) arr[i] /= norm;
  }
  return arr;
}

function randomNormalizedVectors(n, dim) {
  const flat = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const vec = randomFloat32Array(dim);
    normalizeInPlace(vec);
    flat.set(vec, i * dim);
  }
  return flat;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatMs(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} us`;
  if (ms < 1000) return `${ms.toFixed(2)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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
    wasmRaw,
    jsRaw,
    wasmGzip,
    jsGzip,
    totalRaw: wasmRaw + jsRaw,
    totalGzip: wasmGzip + jsGzip,
  };
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

function benchmarkOnce(fn) {
  const start = performance.now();
  const result = fn();
  const elapsed = performance.now() - start;
  return { elapsed, result };
}

function benchmark(fn, warmup = 2, iterations = 10) {
  // Warmup
  for (let i = 0; i < warmup; i++) fn();
  // Measure
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }
  return {
    median: median(times),
    min: Math.min(...times),
    max: Math.max(...times),
    mean: times.reduce((a, b) => a + b, 0) / times.length,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  // Try to load the Node.js WASM module
  let wasm;
  const pkgNodePath = path.join(__dirname, '..', 'pkg-node', 'turboquant_wasm.js');
  try {
    wasm = require(pkgNodePath);
  } catch (e) {
    console.error(`Could not load WASM module from ${pkgNodePath}`);
    console.error('Build it first: wasm-pack build --target nodejs --out-dir pkg-node');
    console.error(`Error: ${e.message}`);
    process.exit(1);
  }

  const { TurboQuantizer, build_index } = wasm;

  console.log('='.repeat(72));
  console.log('  turboquant-wasm Benchmark');
  console.log('='.repeat(72));
  console.log();

  // ---- Configuration ----
  const configs = [
    { dim: 384, bits: 4, label: 'd=384 (MiniLM), 4-bit' },
    { dim: 384, bits: 2, label: 'd=384 (MiniLM), 2-bit' },
    { dim: 768, bits: 4, label: 'd=768 (MPNet), 4-bit' },
    { dim: 1536, bits: 4, label: 'd=1536 (OpenAI), 4-bit' },
  ];

  const nVectorsList = [1000, 5000, 10000];
  const k = 10;
  const seed = 42n;

  for (const config of configs) {
    console.log('-'.repeat(72));
    console.log(`  ${config.label}`);
    console.log('-'.repeat(72));

    // ---- Quantizer creation ----
    const { elapsed: quantizerTime, result: quantizer } = benchmarkOnce(() => {
      return new TurboQuantizer(config.dim, config.bits, seed);
    });
    console.log(`  Quantizer creation:  ${formatMs(quantizerTime)}`);
    console.log(`  Compression ratio:   ${quantizer.compression_ratio.toFixed(2)}x`);
    console.log(`  Expected MSE:        ${quantizer.expected_mse.toFixed(6)}`);
    console.log();

    // ---- Single vector encode/decode ----
    const testVec = randomFloat32Array(config.dim);
    normalizeInPlace(testVec);

    const encodeResult = benchmark(
      () => quantizer.encode(testVec),
      3, 20
    );
    console.log(`  Encode (1 vector):   ${formatMs(encodeResult.median)} median  (min: ${formatMs(encodeResult.min)}, max: ${formatMs(encodeResult.max)})`);

    // Encode + decode roundtrip fidelity
    const encoded = quantizer.encode(testVec);
    const norm = quantizer.encode_norm(testVec);
    const decoded = quantizer.decode(encoded, norm);

    let mse = 0;
    for (let i = 0; i < config.dim; i++) {
      const diff = testVec[i] - decoded[i];
      mse += diff * diff;
    }
    // Total MSE = E[||x - x̂||²], NOT per-coordinate
    console.log(`  Roundtrip MSE:       ${mse.toFixed(6)} (expected: ${quantizer.expected_mse.toFixed(6)})`);
    console.log();

    // ---- Index build + search for various N ----
    for (const n of nVectorsList) {
      // Skip very large configs to keep benchmark manageable
      if (config.dim >= 1536 && n > 5000) {
        console.log(`  [N=${n}] skipped (would take too long for d=${config.dim})`);
        continue;
      }

      console.log(`  --- N = ${n.toLocaleString()} vectors ---`);

      // Generate random normalized vectors
      const vectors = randomNormalizedVectors(n, config.dim);

      // Build index
      const { elapsed: buildTime, result: index } = benchmarkOnce(() => {
        return build_index(quantizer, vectors, n);
      });
      console.log(`  Build index:         ${formatMs(buildTime)}`);
      console.log(`  Index memory:        ${formatBytes(index.memory_bytes)}`);
      console.log(`  Bytes per vector:    ${(index.memory_bytes / n).toFixed(0)} B`);

      // Search
      const queryVec = randomFloat32Array(config.dim);
      normalizeInPlace(queryVec);

      const searchResult = benchmark(
        () => index.search(quantizer, queryVec, k),
        2, 10
      );
      console.log(`  Search (k=${k}):       ${formatMs(searchResult.median)} median  (min: ${formatMs(searchResult.min)}, max: ${formatMs(searchResult.max)})`);

      // Verify search returns correct number of results
      const results = index.search(quantizer, queryVec, k);
      console.log(`  Results returned:    ${results.length} (expected: ${Math.min(k, n)})`);

      // Throughput
      const searchesPerSec = Math.round(1000 / searchResult.median);
      console.log(`  Search throughput:   ~${searchesPerSec.toLocaleString()} queries/sec`);
      console.log();

      // Free the index
      if (typeof index.free === 'function') index.free();
    }

    // Free the quantizer
    if (typeof quantizer.free === 'function') quantizer.free();
    console.log();
  }

  // ---- Bundle size summary ----
  console.log('='.repeat(72));
  console.log('  Bundle Size Summary');
  console.log('='.repeat(72));
  try {
    const bundle = measureBundleSizes();
    console.log(`  .wasm raw:           ${formatBytes(bundle.wasmRaw)}`);
    console.log(`  .js glue raw:        ${formatBytes(bundle.jsRaw)}`);
    console.log(`  Total raw:           ${formatBytes(bundle.totalRaw)}`);
    console.log(`  .wasm gzip (-9):     ${formatBytes(bundle.wasmGzip)}`);
    console.log(`  .js gzip (-9):       ${formatBytes(bundle.jsGzip)}`);
    console.log(`  Total gzip (-9):     ${formatBytes(bundle.totalGzip)}`);
  } catch (e) {
    console.log(`  Could not read pkg-bundler/ files: ${e.message}`);
  }

  console.log();
  console.log('  Note: no head-to-head benchmark against competing libraries');
  console.log('  is committed in this repository yet.');
  console.log();

  // ---- Memory efficiency comparison ----
  console.log('='.repeat(72));
  console.log('  Memory Efficiency (d=384, 4-bit, 10K vectors)');
  console.log('='.repeat(72));
  const d = 384;
  const n = 10000;
  const tqMemory = n * (d + 4);  // u8 indices + f32 norm
  const rawMemory = n * d * 4;    // float32
  console.log(`  turboquant-wasm:     ${formatBytes(tqMemory)} (${(rawMemory / tqMemory).toFixed(1)}x compression)`);
  console.log(`  Uncompressed f32:    ${formatBytes(rawMemory)}`);
  console.log(`  Savings:             ${formatBytes(rawMemory - tqMemory)} (${((1 - tqMemory / rawMemory) * 100).toFixed(0)}% reduction)`);
  console.log();
}

main().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
