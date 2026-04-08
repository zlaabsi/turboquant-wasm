#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use js_sys::Float32Array;
use turboquant_wasm::{TurboQuantizer, CompressedIndex, build_index};

// Run tests in Node.js (not browser) for CI compatibility
// wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_constructor_valid() {
    let q = TurboQuantizer::new(128, 4, 42).unwrap();
    assert_eq!(q.dim(), 128);
    assert_eq!(q.bits(), 4);
}

#[wasm_bindgen_test]
fn test_constructor_rejects_invalid_bits() {
    assert!(TurboQuantizer::new(128, 0, 42).is_err());
    assert!(TurboQuantizer::new(128, 9, 42).is_err());
}

#[wasm_bindgen_test]
fn test_constructor_rejects_invalid_dim() {
    assert!(TurboQuantizer::new(0, 4, 42).is_err());
    assert!(TurboQuantizer::new(1, 4, 42).is_err());
}

#[wasm_bindgen_test]
fn test_encode_decode_roundtrip() {
    let q = TurboQuantizer::new(64, 4, 42).unwrap();
    let mut data = vec![0.0f32; 64];
    for i in 0..64 {
        data[i] = ((i as f32 + 1.0) * 0.1).sin();
    }
    let embedding = Float32Array::from(&data[..]);

    let indices = q.encode(&embedding).unwrap();
    let norm = q.encode_norm(&embedding);
    let decoded = q.decode(&indices, norm).unwrap();

    assert_eq!(decoded.length(), 64);
    assert!(norm > 0.0);

    // Reconstruction error should be bounded
    let dec_vec: Vec<f32> = decoded.to_vec();
    let mse: f32 = data.iter().zip(dec_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / 64.0;
    let signal_power: f32 = data.iter().map(|x| x.powi(2)).sum::<f32>() / 64.0;
    assert!(mse / signal_power < 0.5, "Relative MSE too high: {}", mse / signal_power);
}

#[wasm_bindgen_test]
fn test_encode_dimension_mismatch() {
    let q = TurboQuantizer::new(64, 4, 42).unwrap();
    let wrong_data = vec![1.0f32; 32];
    let embedding = Float32Array::from(&wrong_data[..]);
    assert!(q.encode(&embedding).is_err());
}

#[wasm_bindgen_test]
fn test_deterministic_encoding() {
    let q1 = TurboQuantizer::new(64, 4, 42).unwrap();
    let q2 = TurboQuantizer::new(64, 4, 42).unwrap();
    let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).cos()).collect();
    let emb = Float32Array::from(&data[..]);

    let idx1 = q1.encode(&emb).unwrap();
    let idx2 = q2.encode(&emb).unwrap();

    for i in 0..64 {
        assert_eq!(idx1.get_index(i), idx2.get_index(i));
    }
}

#[wasm_bindgen_test]
fn test_build_index_and_search() {
    let q = TurboQuantizer::new(32, 4, 42).unwrap();
    let n: usize = 50;
    let dim: usize = 32;

    let mut flat_data = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            if j == (i % dim) {
                flat_data.push(1.0f32);
            } else {
                flat_data.push(0.01f32);
            }
        }
    }
    let embeddings = Float32Array::from(&flat_data[..]);
    let index = build_index(&q, &embeddings, n).unwrap();

    assert_eq!(index.n_vectors(), n);
    assert!(index.memory_bytes() > 0);

    let mut query = vec![0.01f32; dim];
    query[0] = 1.0;
    let query_arr = Float32Array::from(&query[..]);
    let results = index.search(&q, &query_arr, 5).unwrap();

    assert_eq!(results.length(), 5);
    assert_eq!(results.get_index(0), 0);
}

#[wasm_bindgen_test]
fn test_compression_ratio() {
    let q4 = TurboQuantizer::new(128, 4, 42).unwrap();
    let ratio4 = q4.compression_ratio();
    assert!((ratio4 - 8.0).abs() < 1.0, "4-bit should be ~8x, got {}", ratio4);

    let q2 = TurboQuantizer::new(128, 2, 42).unwrap();
    let ratio2 = q2.compression_ratio();
    assert!((ratio2 - 16.0).abs() < 2.0, "2-bit should be ~16x, got {}", ratio2);
}

#[wasm_bindgen_test]
fn test_mse_decreases_with_bits() {
    let mse2 = TurboQuantizer::new(128, 2, 42).unwrap().expected_mse();
    let mse3 = TurboQuantizer::new(128, 3, 42).unwrap().expected_mse();
    let mse4 = TurboQuantizer::new(128, 4, 42).unwrap().expected_mse();
    assert!(mse2 > mse3, "mse2={} > mse3={}", mse2, mse3);
    assert!(mse3 > mse4, "mse3={} > mse4={}", mse3, mse4);
}

#[wasm_bindgen_test]
fn test_bits_5_through_8() {
    for bits in 5..=8u8 {
        let q = TurboQuantizer::new(64, bits, 42).unwrap();
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.3).sin()).collect();
        let emb = Float32Array::from(&data[..]);

        let indices = q.encode(&emb).unwrap();
        let norm = q.encode_norm(&emb);
        let decoded = q.decode(&indices, norm).unwrap();

        assert_eq!(decoded.length(), 64);
    }
}

#[wasm_bindgen_test]
fn test_inner_product_estimate() {
    let q = TurboQuantizer::new(64, 4, 42).unwrap();
    let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();
    let emb = Float32Array::from(&data[..]);

    let indices = q.encode(&emb).unwrap();
    let norm = q.encode_norm(&emb);

    // Inner product with self should be close to norm^2
    let score = q.inner_product_estimate(&indices, norm, &emb).unwrap();
    let expected = data.iter().map(|x| x * x).sum::<f32>();
    assert!((score - expected).abs() / expected < 0.3,
        "Self-IP should be close to ||x||^2: got {}, expected {}", score, expected);
}

#[wasm_bindgen_test]
fn test_streaming_add_vector() {
    let q = TurboQuantizer::new(32, 4, 42).unwrap();
    let dim: usize = 32;
    let n: usize = 50;

    // Create empty index and add vectors one by one
    let mut index = CompressedIndex::new_empty(32, 4).unwrap();
    assert_eq!(index.n_vectors(), 0);

    for i in 0..n {
        let mut vec_data = vec![0.01f32; dim];
        vec_data[i % dim] = 1.0;
        let emb = Float32Array::from(&vec_data[..]);
        index.add_vector(&q, &emb).unwrap();
    }

    assert_eq!(index.n_vectors(), n);
    assert!(index.memory_bytes() > 0);

    // Search: query close to vector 0
    let mut query = vec![0.01f32; dim];
    query[0] = 1.0;
    let query_arr = Float32Array::from(&query[..]);
    let results = index.search(&q, &query_arr, 5).unwrap();

    assert_eq!(results.length(), 5);
    // The top result should be vector 0 (strongest match on dim 0)
    assert_eq!(results.get_index(0), 0);
}

#[wasm_bindgen_test]
fn test_add_vectors_batch() {
    let q = TurboQuantizer::new(32, 4, 42).unwrap();
    let dim: usize = 32;
    let n: usize = 50;

    // Build test data
    let mut flat_data = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            if j == (i % dim) {
                flat_data.push(1.0f32);
            } else {
                flat_data.push(0.01f32);
            }
        }
    }
    let embeddings = Float32Array::from(&flat_data[..]);

    // Method 1: build_index (all at once)
    let index_full = build_index(&q, &embeddings, n).unwrap();

    // Method 2: add_vectors (batch append to empty)
    let mut index_batch = CompressedIndex::new_empty(32, 4).unwrap();
    index_batch.add_vectors(&q, &embeddings, n).unwrap();

    assert_eq!(index_batch.n_vectors(), index_full.n_vectors());
    assert_eq!(index_batch.memory_bytes(), index_full.memory_bytes());

    // Both should return the same search results
    let mut query = vec![0.01f32; dim];
    query[0] = 1.0;
    let query_arr = Float32Array::from(&query[..]);

    let results_full = index_full.search(&q, &query_arr, 5).unwrap();
    let results_batch = index_batch.search(&q, &query_arr, 5).unwrap();

    for i in 0..5 {
        assert_eq!(
            results_full.get_index(i), results_batch.get_index(i),
            "Mismatch at position {}: build_index={} vs add_vectors={}",
            i, results_full.get_index(i), results_batch.get_index(i)
        );
    }
}

#[wasm_bindgen_test]
fn test_search_sorts_results_when_k_gte_n_vectors() {
    let q = TurboQuantizer::new(2, 8, 42).unwrap();
    let data = Float32Array::from(&[
        0.0f32, 1.0f32,
        1.0f32, 0.0f32,
        0.5f32, 0.5f32,
    ][..]);
    let index = build_index(&q, &data, 3).unwrap();
    let query = Float32Array::from(&[1.0f32, 0.0f32][..]);

    let results = index.search(&q, &query, 10).unwrap();

    assert_eq!(results.length(), 3);
    assert_eq!(results.get_index(0), 1);
    assert_eq!(results.get_index(1), 2);
    assert_eq!(results.get_index(2), 0);
}

#[wasm_bindgen_test]
fn test_search_rejects_quantizer_bits_mismatch() {
    let q4 = TurboQuantizer::new(32, 4, 42).unwrap();
    let q2 = TurboQuantizer::new(32, 2, 42).unwrap();
    let mut index = CompressedIndex::new_empty(32, 4).unwrap();
    let embedding = Float32Array::from(&[
        1.0f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
        0.01f32, 0.01f32, 0.01f32, 0.01f32,
    ][..]);

    index.add_vector(&q4, &embedding).unwrap();
    assert!(index.search(&q2, &embedding, 1).is_err());
}
