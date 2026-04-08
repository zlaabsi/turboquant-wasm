export interface QuantizerOptions {
  dim: number;
  bits?: number;
  seed?: bigint | number;
}

export interface SearchResult {
  index: number;
  score: number;
}

export interface EncodedVector {
  indices: Uint8Array;
  norm: number;
}

export declare class Quantizer {
  constructor(inner: unknown, dim: number, bits: number);
  readonly dim: number;
  readonly bits: number;
  encode(embedding: Float32Array): EncodedVector;
  decode(indices: Uint8Array, norm: number): Float32Array;
  innerProduct(indices: Uint8Array, norm: number, query: Float32Array): number;
  buildIndex(embeddings: Float32Array, n: number): Index;
  createEmptyIndex(): Index;
  get compressionRatio(): number;
  get expectedMse(): number;
  free(): void;
}

export declare class Index {
  constructor(inner: unknown, quantizer: Quantizer);
  static load(data: Uint8Array, quantizer: Quantizer): Index;
  get size(): number;
  get memoryBytes(): number;
  addVector(embedding: Float32Array): void;
  addVectors(embeddings: Float32Array, n: number): void;
  search(query: Float32Array, k?: number): number[];
  save(): Uint8Array;
  free(): void;
}

export declare function createQuantizer(options: QuantizerOptions): Promise<Quantizer>;
export declare function flattenEmbeddings(embeddings: number[][]): Float32Array;
