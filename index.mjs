import init, {
  TurboQuantizer as WasmQuantizer,
  CompressedIndex as WasmIndex,
  build_index,
} from "./pkg/turboquant_wasm.js";

let initPromise = null;

function ensureInit() {
  if (!initPromise) {
    initPromise = init();
  }
  return initPromise;
}

function normalizeSeed(seed = 42n) {
  return typeof seed === "bigint" ? seed : BigInt(seed);
}

export class Quantizer {
  constructor(inner, dim, bits) {
    this.inner = inner;
    this.dim = dim;
    this.bits = bits;
  }

  encode(embedding) {
    const indices = this.inner.encode(embedding);
    const norm = this.inner.encode_norm(embedding);
    return { indices, norm };
  }

  decode(indices, norm) {
    return this.inner.decode(indices, norm);
  }

  innerProduct(indices, norm, query) {
    return this.inner.inner_product_estimate(indices, norm, query);
  }

  buildIndex(embeddings, n) {
    return new Index(build_index(this.inner, embeddings, n), this);
  }

  createEmptyIndex() {
    return new Index(WasmIndex.new_empty(this.dim, this.bits), this);
  }

  get compressionRatio() {
    return this.inner.compression_ratio;
  }

  get expectedMse() {
    return this.inner.expected_mse;
  }

  free() {
    this.inner.free();
  }
}

export class Index {
  constructor(inner, quantizer) {
    this.inner = inner;
    this.quantizer = quantizer;
  }

  static load(data, quantizer) {
    return new Index(WasmIndex.load(data), quantizer);
  }

  get size() {
    return this.inner.n_vectors;
  }

  get memoryBytes() {
    return this.inner.memory_bytes;
  }

  addVector(embedding) {
    this.inner.add_vector(this.quantizer.inner, embedding);
  }

  addVectors(embeddings, n) {
    this.inner.add_vectors(this.quantizer.inner, embeddings, n);
  }

  search(query, k = 10) {
    return Array.from(this.inner.search(this.quantizer.inner, query, k));
  }

  save() {
    return this.inner.save();
  }

  free() {
    this.inner.free();
  }
}

export async function createQuantizer(options) {
  await ensureInit();

  const bits = options.bits ?? 4;
  const seed = normalizeSeed(options.seed);
  const inner = new WasmQuantizer(options.dim, bits, seed);
  return new Quantizer(inner, options.dim, bits);
}

export function flattenEmbeddings(embeddings) {
  const dim = embeddings[0].length;
  const flat = new Float32Array(embeddings.length * dim);
  for (let i = 0; i < embeddings.length; i += 1) {
    flat.set(embeddings[i], i * dim);
  }
  return flat;
}
