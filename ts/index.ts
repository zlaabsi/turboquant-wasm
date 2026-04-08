/**
 * turboquant-wasm — TurboQuant vector quantization in the browser.
 *
 * High-level TypeScript API wrapping the Rust WASM core.
 * Implements Algorithm 1 (TurboQuant_mse) from arXiv:2504.19874.
 *
 * @example
 * ```ts
 * import { createQuantizer, buildIndex } from '@zlaabsi/turboquant-wasm';
 *
 * const quantizer = await createQuantizer({ dim: 1536, bits: 4 });
 * const index = await quantizer.index(embeddings);
 * const results = index.search(queryVector, 10);
 * ```
 *
 * @example Streaming API (incremental index building)
 * ```ts
 * const quantizer = await createQuantizer({ dim: 768, bits: 2 });
 * const index = quantizer.createEmptyIndex();
 * for (const embedding of stream) {
 *   index.addVector(quantizer, embedding);
 * }
 * const results = index.search(queryVector, 5);
 * ```
 */

import init, {
  TurboQuantizer as WasmQuantizer,
  CompressedIndex as WasmIndex,
  build_index,
} from '../pkg/turboquant_wasm.js';

export interface QuantizerOptions {
  /** Embedding dimension (e.g. 384, 768, 1536) */
  dim: number;
  /** Bits per coordinate (1-8, default: 4) */
  bits?: number;
  /** Random seed for rotation matrix (default: 42) */
  seed?: bigint;
}

export interface SearchResult {
  /** Index of the vector in the original array */
  index: number;
  /** Approximate inner product score */
  score: number;
}

export interface EncodedVector {
  /** Quantized coordinate indices (packed) */
  indices: Uint8Array;
  /** L2 norm of the original vector */
  norm: number;
}

export class Quantizer {
  /** @internal */
  private inner: WasmQuantizer;
  readonly dim: number;
  readonly bits: number;

  constructor(inner: WasmQuantizer, dim: number, bits: number) {
    this.inner = inner;
    this.dim = dim;
    this.bits = bits;
  }

  /**
   * Compress a single embedding vector.
   * Returns the quantized indices and original norm.
   */
  encode(embedding: Float32Array): EncodedVector {
    const indices = this.inner.encode(embedding);
    const norm = this.inner.encode_norm(embedding);
    return { indices, norm };
  }

  /**
   * Decompress indices back to an approximate float32 vector.
   * @param indices - Quantized coordinate indices from encode()
   * @param norm - L2 norm from encode()
   */
  decode(indices: Uint8Array, norm: number): Float32Array {
    return this.inner.decode(indices, norm);
  }

  /**
   * Estimate inner product between a compressed vector and a full-precision query.
   * Uses the rotated-domain optimization: O(d) per vector instead of O(d^2).
   */
  innerProduct(indices: Uint8Array, norm: number, query: Float32Array): number {
    return this.inner.inner_product_estimate(indices, norm, query);
  }

  /**
   * Build a compressed index from a batch of embeddings.
   * @param embeddings - Flat Float32Array of n x dim values (row-major)
   * @param n - Number of vectors
   */
  buildIndex(embeddings: Float32Array, n: number): Index {
    const wasmIndex = build_index(this.inner, embeddings, n);
    return new Index(wasmIndex, this);
  }

  /**
   * Create an empty index for incremental/streaming vector insertion.
   * Use with Index.addVector() or Index.addVectors() to populate.
   */
  createEmptyIndex(): Index {
    const wasmIndex = WasmIndex.new_empty(this.dim, this.bits);
    return new Index(wasmIndex, this);
  }

  /** Compression ratio (original float32 size / compressed size) */
  get compressionRatio(): number {
    return this.inner.compression_ratio;
  }

  /** Expected MSE for unit vectors (Theorem 1 from the paper) */
  get expectedMse(): number {
    return this.inner.expected_mse;
  }

  /** Release WASM memory. Call when done with this quantizer. */
  free(): void {
    this.inner.free();
  }
}

export class Index {
  /** @internal */
  private inner: WasmIndex;
  /** @internal */
  private quantizer: Quantizer;

  constructor(inner: WasmIndex, quantizer: Quantizer) {
    this.inner = inner;
    this.quantizer = quantizer;
  }

  /** Number of vectors in the index */
  get size(): number {
    return this.inner.n_vectors;
  }

  /** Memory usage of the compressed index in bytes */
  get memoryBytes(): number {
    return this.inner.memory_bytes;
  }

  /**
   * Add a single vector to the index (streaming/incremental).
   *
   * The vector is encoded (rotated, quantized, packed) and appended
   * to the index storage. No rebuild needed.
   *
   * @param embedding - Float32Array of length dim
   */
  addVector(embedding: Float32Array): void {
    this.inner.add_vector(
      (this.quantizer as any).inner,
      embedding
    );
  }

  /**
   * Add multiple vectors to the index in batch.
   *
   * More efficient than calling addVector() in a loop because it
   * avoids per-call WASM boundary overhead.
   *
   * @param embeddings - Flat Float32Array of n x dim values (row-major)
   * @param n - Number of vectors
   */
  addVectors(embeddings: Float32Array, n: number): void {
    this.inner.add_vectors(
      (this.quantizer as any).inner,
      embeddings,
      n
    );
  }

  /**
   * Search for k nearest neighbors by approximate inner product.
   * @param query - Float32Array of length dim (does not need to be unit-normalized)
   * @param k - Number of results to return (default: 10)
   * @returns Array of vector indices sorted by descending score
   */
  search(query: Float32Array, k: number = 10): number[] {
    const result = this.inner.search(
      (this.quantizer as any).inner,
      query,
      k
    );
    return Array.from(result);
  }

  /** Release WASM memory. Call when done with this index. */
  free(): void {
    this.inner.free();
  }
}

/**
 * Initialize the WASM module and create a quantizer.
 * Call this once at startup. Subsequent calls reuse the initialized module.
 *
 * @param options - Quantizer configuration (dim, bits, seed)
 */
export async function createQuantizer(options: QuantizerOptions): Promise<Quantizer> {
  await init();

  const bits = options.bits ?? 4;
  const seed = options.seed ?? 42n;

  const inner = new WasmQuantizer(options.dim, bits, seed);
  return new Quantizer(inner, options.dim, bits);
}

/**
 * Flatten an array of embedding arrays into a single Float32Array.
 * Useful for passing to buildIndex() or addVectors().
 */
export function flattenEmbeddings(embeddings: number[][]): Float32Array {
  const dim = embeddings[0].length;
  const flat = new Float32Array(embeddings.length * dim);
  for (let i = 0; i < embeddings.length; i++) {
    flat.set(embeddings[i], i * dim);
  }
  return flat;
}
