//! TurboQuant WASM — Algorithm 1 (TurboQuant_mse) for browser-side vector search.
//!
//! Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026 (arXiv:2504.19874)
//!
//! Implements Algorithm 1 only (MSE-optimal). QJL (Algorithm 2) is omitted for WASM
//! because the QJL projection matrix S ∈ R^{d×d} doubles the memory footprint, which
//! is prohibitive in browser environments for d ≥ 768. For vector search at bits ≥ 3,
//! Algorithm 1 alone gives excellent recall (the MSE bias is small).
//!
//! KEY OPTIMIZATION — Search in the rotated domain:
//!   Since Π is orthogonal, ⟨x, q⟩ = ⟨Πx, Πq⟩.
//!   We store quantized codes of Πx. At search time, we rotate q ONCE to get Πq,
//!   then compute approximate dot products directly as:
//!     score ≈ ‖x‖ · Σ_j c_{idx_j} · (Πq)_j
//!   This is O(d) per vector instead of O(d²) (no inverse rotation needed).

use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Uint8Array};

// ===========================================================================
// SIMD-accelerated dot product for WASM (used in query rotation + encode)
//
// WASM SIMD128 is supported in Chrome 91+, Firefox 89+, Safari 16.4+, Node 16.4+.
// When target-feature=+simd128 is set (via .cargo/config.toml), this compiles
// to native v128 instructions. Otherwise, scalar fallback is used.
// ===========================================================================

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

/// SIMD f32x4 dot product of two equal-length slices.
#[cfg(target_feature = "simd128")]
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let simd_len = len & !3;

    let mut acc = f32x4_splat(0.0);
    let mut i = 0;
    while i < simd_len {
        let va = unsafe { v128_load(a.as_ptr().add(i) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(i) as *const v128) };
        acc = f32x4_add(acc, f32x4_mul(va, vb));
        i += 4;
    }

    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Scalar fallback dot product.
#[cfg(not(target_feature = "simd128"))]
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}

/// Matrix-vector multiply: result[i] = dot(matrix[i*dim..(i+1)*dim], vec)
#[inline]
fn mat_vec_mul(matrix: &[f32], vec: &[f32], dim: usize) -> Vec<f32> {
    let n_rows = matrix.len() / dim;
    let mut out = vec![0.0f32; n_rows];
    for i in 0..n_rows {
        out[i] = dot_f32(&matrix[i * dim..(i + 1) * dim], vec);
    }
    out
}

// ===========================================================================
// Lloyd-Max centroids for N(0,1) — cross-validated with paper
//
// Paper Section 1.3:
//   b=1: ±√(2/π)/√d → N(0,1) centroids ±0.79788  ✓
//   b=2: ±0.453/√d, ±1.51/√d → ±0.45278, ±1.51042  ✓
//
// MSE (Theorem 1): b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009  ✓
//
// At runtime, centroids are scaled by 1/√d to match the distribution
// of coordinates of Π·x for x ∈ S^{d-1}.
// ===========================================================================

const CENTROIDS_1BIT: [f32; 2] = [-0.7978846, 0.7978846];

const CENTROIDS_2BIT: [f32; 4] = [-1.5104176, -0.4527800, 0.4527800, 1.5104176];

const CENTROIDS_3BIT: [f32; 8] = [
    -2.1519775, -1.3439093, -0.7560052, -0.2451210,
     0.2451210,  0.7560052,  1.3439093,  2.1519775,
];

const CENTROIDS_4BIT: [f32; 16] = [
    -2.7326369, -2.0690096, -1.6180040, -1.2562091,
    -0.9423403, -0.6568096, -0.3881420, -0.1284186,
     0.1284186,  0.3881420,  0.6568096,  0.9423403,
     1.2562091,  1.6180040,  2.0690096,  2.7326369,
];

const CENTROIDS_5BIT: [f32; 32] = [
    -3.25552, -2.68521, -2.31140, -2.02214, -1.78054, -1.56958, -1.37986, -1.20562,
    -1.04301, -0.88932, -0.74252, -0.60103, -0.46358, -0.32911, -0.19667, -0.06543,
     0.06543,  0.19667,  0.32911,  0.46358,  0.60103,  0.74252,  0.88932,  1.04301,
     1.20562,  1.37986,  1.56958,  1.78054,  2.02214,  2.31140,  2.68521,  3.25552,
];

const CENTROIDS_6BIT: [f32; 64] = [
    -3.60579, -3.08549, -2.75084, -2.49669, -2.28850, -2.11042, -1.95377, -1.81328,
    -1.68547, -1.56795, -1.45897, -1.35723, -1.26171, -1.17161, -1.08629, -1.00519,
    -0.92788, -0.85395, -0.78306, -0.71491, -0.64923, -0.58575, -0.52425, -0.46450,
    -0.40630, -0.34945, -0.29376, -0.23904, -0.18511, -0.13178, -0.07890, -0.02627,
     0.02627,  0.07890,  0.13178,  0.18511,  0.23904,  0.29376,  0.34945,  0.40630,
     0.46450,  0.52425,  0.58575,  0.64923,  0.71491,  0.78306,  0.85395,  0.92788,
     1.00519,  1.08629,  1.17161,  1.26171,  1.35723,  1.45897,  1.56795,  1.68547,
     1.81328,  1.95377,  2.11042,  2.28850,  2.49669,  2.75084,  3.08549,  3.60579,
];

const CENTROIDS_7BIT: [f32; 128] = [
    -3.83523, -3.34311, -3.02949, -2.79327, -2.60133, -2.43844, -2.29631, -2.16988,
    -2.05583, -1.95187, -1.85634, -1.76799, -1.68586, -1.60920, -1.53739, -1.46993,
    -1.40639, -1.34642, -1.28969, -1.23595, -1.18495, -1.13647, -1.09031, -1.04631,
    -1.00430, -0.96412, -0.92564, -0.88874, -0.85329, -0.81918, -0.78630, -0.75457,
    -0.72389, -0.69418, -0.66535, -0.63735, -0.61009, -0.58352, -0.55758, -0.53222,
    -0.50738, -0.48302, -0.45910, -0.43559, -0.41244, -0.38962, -0.36710, -0.34486,
    -0.32286, -0.30110, -0.27953, -0.25815, -0.23694, -0.21588, -0.19494, -0.17412,
    -0.15341, -0.13278, -0.11223, -0.09174, -0.07130, -0.05090, -0.03053, -0.01017,
     0.01017,  0.03053,  0.05090,  0.07130,  0.09174,  0.11223,  0.13278,  0.15341,
     0.17412,  0.19494,  0.21588,  0.23694,  0.25815,  0.27953,  0.30110,  0.32286,
     0.34486,  0.36710,  0.38962,  0.41244,  0.43559,  0.45910,  0.48302,  0.50738,
     0.53222,  0.55758,  0.58352,  0.61009,  0.63735,  0.66535,  0.69418,  0.72389,
     0.75457,  0.78630,  0.81918,  0.85329,  0.88874,  0.92564,  0.96412,  1.00430,
     1.04631,  1.09031,  1.13647,  1.18495,  1.23595,  1.28969,  1.34642,  1.40639,
     1.46993,  1.53739,  1.60920,  1.68586,  1.76799,  1.85634,  1.95187,  2.05583,
     2.16988,  2.29631,  2.43844,  2.60133,  2.79327,  3.02949,  3.34311,  3.83523,
];

const CENTROIDS_8BIT: [f32; 256] = [
    -4.03524, -3.56536, -3.26791, -3.04519, -2.86519, -2.71324, -2.58133, -2.46458,
    -2.35979, -2.26475, -2.17785, -2.09789, -2.02395, -1.95528, -1.89130, -1.83151,
    -1.77550, -1.72293, -1.67348, -1.62691, -1.58296, -1.54144, -1.50214, -1.46491,
    -1.42959, -1.39602, -1.36408, -1.33365, -1.30461, -1.27686, -1.25030, -1.22483,
    -1.20039, -1.17687, -1.15423, -1.13238, -1.11127, -1.09084, -1.07103, -1.05181,
    -1.03312, -1.01493, -0.99719, -0.97988, -0.96296, -0.94640, -0.93019, -0.91429,
    -0.89868, -0.88335, -0.86828, -0.85345, -0.83885, -0.82446, -0.81028, -0.79629,
    -0.78247, -0.76883, -0.75535, -0.74203, -0.72885, -0.71581, -0.70291, -0.69014,
    -0.67749, -0.66496, -0.65254, -0.64023, -0.62803, -0.61593, -0.60392, -0.59200,
    -0.58018, -0.56844, -0.55678, -0.54521, -0.53371, -0.52228, -0.51093, -0.49965,
    -0.48843, -0.47728, -0.46619, -0.45516, -0.44419, -0.43328, -0.42242, -0.41161,
    -0.40085, -0.39014, -0.37948, -0.36886, -0.35828, -0.34775, -0.33725, -0.32680,
    -0.31638, -0.30600, -0.29565, -0.28534, -0.27505, -0.26480, -0.25458, -0.24438,
    -0.23421, -0.22406, -0.21394, -0.20384, -0.19377, -0.18371, -0.17367, -0.16365,
    -0.15365, -0.14366, -0.13369, -0.12373, -0.11379, -0.10386, -0.09393, -0.08402,
    -0.07411, -0.06422, -0.05433, -0.04444, -0.03456, -0.02468, -0.01481, -0.00494,
     0.00494,  0.01481,  0.02468,  0.03456,  0.04444,  0.05433,  0.06422,  0.07411,
     0.08402,  0.09393,  0.10386,  0.11379,  0.12373,  0.13369,  0.14366,  0.15365,
     0.16365,  0.17367,  0.18371,  0.19377,  0.20384,  0.21394,  0.22406,  0.23421,
     0.24438,  0.25458,  0.26480,  0.27505,  0.28534,  0.29565,  0.30600,  0.31638,
     0.32680,  0.33725,  0.34775,  0.35828,  0.36886,  0.37948,  0.39014,  0.40085,
     0.41161,  0.42242,  0.43328,  0.44419,  0.45516,  0.46619,  0.47728,  0.48843,
     0.49965,  0.51093,  0.52228,  0.53371,  0.54521,  0.55678,  0.56844,  0.58018,
     0.59200,  0.60392,  0.61593,  0.62803,  0.64023,  0.65254,  0.66496,  0.67749,
     0.69014,  0.70291,  0.71581,  0.72885,  0.74203,  0.75535,  0.76883,  0.78247,
     0.79629,  0.81028,  0.82446,  0.83885,  0.85345,  0.86828,  0.88335,  0.89868,
     0.91429,  0.93019,  0.94640,  0.96296,  0.97988,  0.99719,  1.01493,  1.03312,
     1.05181,  1.07103,  1.09084,  1.11127,  1.13238,  1.15423,  1.17687,  1.20039,
     1.22483,  1.25030,  1.27686,  1.30461,  1.33365,  1.36408,  1.39602,  1.42959,
     1.46491,  1.50214,  1.54144,  1.58296,  1.62691,  1.67348,  1.72293,  1.77550,
     1.83151,  1.89130,  1.95528,  2.02395,  2.09789,  2.17785,  2.26475,  2.35979,
     2.46458,  2.58133,  2.71324,  2.86519,  3.04519,  3.26791,  3.56536,  4.03524,
];

/// Get N(0,1) centroids for a given bit-width (1–8).
fn get_n01_centroids(bits: u8) -> &'static [f32] {
    match bits {
        1 => &CENTROIDS_1BIT,
        2 => &CENTROIDS_2BIT,
        3 => &CENTROIDS_3BIT,
        4 => &CENTROIDS_4BIT,
        5 => &CENTROIDS_5BIT,
        6 => &CENTROIDS_6BIT,
        7 => &CENTROIDS_7BIT,
        8 => &CENTROIDS_8BIT,
        _ => unreachable!("bits validated in constructor"),
    }
}

/// Compute decision boundaries (midpoints of consecutive centroids).
#[inline]
fn compute_boundaries(centroids: &[f32]) -> Vec<f32> {
    centroids
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect()
}

/// Quantize a scalar to the nearest centroid using binary search.
/// Returns index in [0, n_centroids).
#[inline(always)]
fn quantize_scalar(value: f32, boundaries: &[f32]) -> u8 {
    boundaries.partition_point(|&b| b <= value) as u8
}

// ===========================================================================
// Deterministic PRNG — xoshiro256** (NOT the same as NumPy's PCG64 default_rng)
// ===========================================================================

struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 to expand single seed into 4×64-bit state
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *si = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64); // uniform [0,1)
        let u2 = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        let u1 = u1.max(1e-15);
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

// ===========================================================================
// Haar-distributed orthogonal matrix
//
// Paper: "We can generate Π by applying QR decomposition on a random
// matrix with i.i.d Normal entries."
//
// Two paths:
//   dim < 512  → Modified Gram-Schmidt (fast, adequate orthogonality)
//   dim >= 512 → Householder QR + Mezzadri correction (numerically stable)
//
// Gram-Schmidt loses orthogonality at O(d * eps_machine) where
// eps_machine ~= 1.2e-7 for f32. At dim=1536 this gives ~1.8e-4 error.
// Householder reflections maintain machine-precision orthogonality.
// ===========================================================================

/// Householder QR: produces a Haar-distributed orthogonal matrix.
///
/// Generates a dim x dim Gaussian matrix, computes QR via Householder
/// reflections (storing vectors in the lower triangle), reconstructs Q,
/// and applies the Mezzadri sign correction.
///
/// Returns dim x dim row-major orthogonal matrix.
fn householder_haar_matrix(dim: usize, rng: &mut Xoshiro256) -> Vec<f32> {
    // Generate random Gaussian matrix in column-major order.
    // Column-major: a[col * dim + row], so we fill column by column.
    // PRNG consumption order: same as filling dim*dim entries sequentially.
    let mut a = vec![0.0f32; dim * dim];
    for v in a.iter_mut() {
        *v = rng.next_normal();
    }

    // Store Householder vectors and diagonal of R
    let mut tau = vec![0.0f32; dim]; // 2 / (v^T v) for each reflection
    let mut diag_r = vec![0.0f32; dim];

    for k in 0..dim {
        let col_k = k * dim;

        // Compute ||x|| for a[k..dim, k] in f64
        let mut norm_sq = 0.0f64;
        for i in k..dim {
            let val = a[col_k + i] as f64;
            norm_sq += val * val;
        }
        let x_norm = norm_sq.sqrt();

        if x_norm < 1e-30 {
            diag_r[k] = 0.0;
            tau[k] = 0.0;
            continue;
        }

        // alpha = -sign(a[k,k]) * ||x||
        let sign = if a[col_k + k] >= 0.0 { 1.0f64 } else { -1.0f64 };
        let alpha = -sign * x_norm;
        diag_r[k] = alpha as f32;

        // v = x; v[0] -= alpha
        // Then tau = 2 / (v^T v)
        // Store v in-place in a[k:, k] (the lower part of column k).
        a[col_k + k] -= alpha as f32;

        let mut v_norm_sq = 0.0f64;
        for i in k..dim {
            let val = a[col_k + i] as f64;
            v_norm_sq += val * val;
        }

        if v_norm_sq < 1e-60 {
            tau[k] = 0.0;
            continue;
        }

        tau[k] = (2.0 / v_norm_sq) as f32;

        // Apply H_k to remaining columns: a[:, j] -= tau * v * (v^T a[:, j])
        // Only the submatrix a[k:, k+1:] is affected.
        for j in (k + 1)..dim {
            let col_j = j * dim;
            let mut dot = 0.0f64;
            for i in k..dim {
                dot += a[col_k + i] as f64 * a[col_j + i] as f64;
            }
            let scale = tau[k] as f64 * dot;
            for i in k..dim {
                a[col_j + i] -= (scale * a[col_k + i] as f64) as f32;
            }
        }
    }

    // Reconstruct Q by accumulating Householder reflections backwards.
    // Q = H_0 * H_1 * ... * H_{d-1}
    // Start with Q = I, apply H_{d-1}, H_{d-2}, ..., H_0
    // Each H_k = I - tau_k * v_k * v_k^T
    // v_k is stored in a[k:, k] (lower triangle of the modified A).
    let mut q = vec![0.0f32; dim * dim];
    // Initialize Q as identity (column-major)
    for i in 0..dim {
        q[i * dim + i] = 1.0;
    }

    // Apply reflections in reverse order
    for k in (0..dim).rev() {
        if tau[k] == 0.0 {
            continue;
        }
        let col_k = k * dim;

        // Apply H_k to columns k..dim of Q:
        // Q[:, j] -= tau * v * (v^T * Q[:, j]) for the subrows k..dim
        for j in k..dim {
            let q_col_j = j * dim;
            let mut dot = 0.0f64;
            for i in k..dim {
                dot += a[col_k + i] as f64 * q[q_col_j + i] as f64;
            }
            let scale = tau[k] as f64 * dot;
            for i in k..dim {
                q[q_col_j + i] -= (scale * a[col_k + i] as f64) as f32;
            }
        }
    }

    // Mezzadri correction: multiply column k of Q by sign(diag_r[k])
    // This ensures Q is Haar-distributed on O(d), not just on SO(d).
    for k in 0..dim {
        let sign = if diag_r[k] >= 0.0 { 1.0f32 } else { -1.0f32 };
        if sign < 0.0 {
            let col_k = k * dim;
            for i in 0..dim {
                q[col_k + i] = -q[col_k + i];
            }
        }
    }

    // Convert from column-major to row-major
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = q[j * dim + i];
        }
    }

    result
}

fn gram_schmidt_orthogonal_matrix(m: &mut [f32], dim: usize) {
    // Modified Gram-Schmidt (row-by-row orthogonalization)
    for i in 0..dim {
        // Normalize row i
        let ri = i * dim;
        let mut norm_sq = 0.0f32;
        for j in 0..dim {
            norm_sq += m[ri + j] * m[ri + j];
        }
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-10);
        for j in 0..dim {
            m[ri + j] *= inv_norm;
        }

        // Project out row i from all subsequent rows
        for k in (i + 1)..dim {
            let rk = k * dim;
            let mut dot = 0.0f32;
            for j in 0..dim {
                dot += m[ri + j] * m[rk + j];
            }
            for j in 0..dim {
                m[rk + j] -= dot * m[ri + j];
            }
        }
    }
}

fn haar_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xoshiro256::from_seed(seed);

    if dim >= 512 {
        return householder_haar_matrix(dim, &mut rng);
    }

    // Small dims: MGS is fast and accurate enough in f32
    let mut m = vec![0.0f32; dim * dim];
    for v in m.iter_mut() {
        *v = rng.next_normal();
    }
    gram_schmidt_orthogonal_matrix(&mut m, dim);
    m
}

// ===========================================================================
// TurboQuantizer — Algorithm 1 (TurboQuant_mse) exposed to WASM
// ===========================================================================

#[wasm_bindgen]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    /// Π — (dim × dim) row-major orthogonal rotation matrix
    rotation: Vec<f32>,
    /// Scaled centroids: N(0,1) centroids / √dim. These live in [-1, 1].
    centroids: Vec<f32>,
    /// Midpoint boundaries between consecutive centroids.
    boundaries: Vec<f32>,
    /// 1/√dim — stored for potential serialization / future use
    #[allow(dead_code)]
    inv_sqrt_dim: f32,
    #[allow(dead_code)]
    seed: u64,
}

#[wasm_bindgen]
impl TurboQuantizer {
    /// Create a new TurboQuantizer (Algorithm 1).
    ///
    /// - `dim`: embedding dimension (e.g. 384, 768, 1536)
    /// - `bits`: bits per coordinate (1–4, extendable to 8)
    /// - `seed`: random seed for the rotation matrix Π (deterministic)
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<TurboQuantizer, JsError> {
        if bits < 1 || bits > 8 {
            return Err(JsError::new("bits must be in [1, 8]"));
        }
        if dim < 2 {
            return Err(JsError::new("dim must be ≥ 2"));
        }

        let inv_sqrt_dim = 1.0 / (dim as f32).sqrt();

        // Scale N(0,1) centroids by 1/√d (paper Section 1.1)
        let n01 = get_n01_centroids(bits);
        let centroids: Vec<f32> = n01.iter().map(|&c| c * inv_sqrt_dim).collect();
        let boundaries = compute_boundaries(&centroids);
        let rotation = haar_orthogonal_matrix(dim, seed);

        Ok(TurboQuantizer {
            dim,
            bits,
            rotation,
            centroids,
            boundaries,
            inv_sqrt_dim,
            seed,
        })
    }

    /// Encode a single vector. Returns (indices, norm).
    ///
    /// Algorithm 1 steps:
    ///   1. ‖x‖ → stored as norm
    ///   2. x̂ = x / ‖x‖
    ///   3. y = Π · x̂   (each y_j ∈ [-1, 1])
    ///   4. idx_j = argmin_k |y_j − c_k|
    #[wasm_bindgen]
    pub fn encode(&self, embedding: &Float32Array) -> Result<Uint8Array, JsError> {
        let x = embedding.to_vec();
        if x.len() != self.dim {
            return Err(JsError::new(&format!(
                "Expected dim={}, got {}", self.dim, x.len()
            )));
        }

        // Compute norm, normalize
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

        // Normalize in place
        let x_hat: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();

        // Rotate: y = Π · x̂ (SIMD-accelerated matrix-vector multiply)
        let y = mat_vec_mul(&self.rotation, &x_hat, self.dim);

        // Quantize each coordinate
        let mut indices = vec![0u8; self.dim];
        for i in 0..self.dim {
            indices[i] = quantize_scalar(y[i], &self.boundaries);
        }

        Ok(Uint8Array::from(&indices[..]))
    }

    /// Extract the norm of an embedding vector.
    #[wasm_bindgen]
    pub fn encode_norm(&self, embedding: &Float32Array) -> f32 {
        embedding.to_vec().iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Decode: look up centroids, apply Π^T, scale by norm.
    ///
    /// DEQUANT_mse (Algorithm 1, lines 9–11):
    ///   ỹ_j = c_{idx_j}
    ///   x̃ = Π^T · ỹ  (then multiply by norm)
    #[wasm_bindgen]
    pub fn decode(&self, indices: &Uint8Array, norm: f32) -> Result<Float32Array, JsError> {
        let idx = indices.to_vec();
        if idx.len() != self.dim {
            return Err(JsError::new(&format!(
                "Expected dim={}, got {}", self.dim, idx.len()
            )));
        }

        // Look up centroids: ỹ_j = c_{idx_j}
        let y_hat: Vec<f32> = idx.iter().map(|&i| {
            self.centroids[(i as usize).min(self.centroids.len() - 1)]
        }).collect();

        // x̃ = Π^T · ỹ  (Π^T[j][i] = Π[i][j], column j of Π = row j of Π^T)
        // x̃_j = Σ_i Π[i][j] · ỹ_i
        let mut x_hat = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let mut sum = 0.0f32;
            for i in 0..self.dim {
                sum += self.rotation[i * self.dim + j] * y_hat[i];
            }
            x_hat[j] = sum * norm;
        }

        Ok(Float32Array::from(&x_hat[..]))
    }

    /// Estimate inner product between a compressed vector and a full-precision query.
    ///
    /// Uses the rotated-domain optimization:
    ///   score = norm * Σ_j centroid[idx_j] * (Π·q)_j
    #[wasm_bindgen]
    pub fn inner_product_estimate(
        &self,
        indices: &Uint8Array,
        norm: f32,
        query: &Float32Array,
    ) -> Result<f32, JsError> {
        let idx = indices.to_vec();
        let q = query.to_vec();
        if idx.len() != self.dim || q.len() != self.dim {
            return Err(JsError::new(&format!(
                "Dimension mismatch: expected {}, got indices={} query={}",
                self.dim, idx.len(), q.len()
            )));
        }
        // Rotate query: q_rot = Π · q (SIMD-accelerated)
        let q_rot = mat_vec_mul(&self.rotation, &q, self.dim);
        // Score = norm * Σ_j centroid[idx_j] * q_rot_j
        let mut score = 0.0f32;
        for j in 0..self.dim {
            let centroid = self.centroids[(idx[j] as usize).min(self.centroids.len() - 1)];
            score += centroid * q_rot[j];
        }
        Ok(score * norm)
    }

    /// Get the expected MSE for unit vectors (Theorem 1).
    #[wasm_bindgen(getter)]
    pub fn expected_mse(&self) -> f32 {
        match self.bits {
            1 => 0.3634,
            2 => 0.1175,
            3 => 0.0302,
            4 => 0.0095,
            5 => 0.00252,
            6 => 0.000699,
            7 => 0.000252,
            8 => 0.000100,
            b => {
                let sqrt3_pi_over2 = (3.0f32).sqrt() * std::f32::consts::PI / 2.0;
                sqrt3_pi_over2 / (4.0f32).powi(b as i32)
            }
        }
    }

    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f32 {
        // Original: dim × 4 bytes (float32)
        // Actual storage: ceil(dim × bits / 8) bytes (bit-packed) + 4 bytes (f32 norm)
        let original = (self.dim * 4) as f32;
        let packed_bytes = ((self.dim * self.bits as usize + 7) / 8) as f32;
        let actual = packed_bytes + 4.0;
        original / actual
    }

    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize { self.dim }

    #[wasm_bindgen(getter)]
    pub fn bits(&self) -> u8 { self.bits }
}

// ===========================================================================
// Bit packing — pack quantization indices to minimize memory
// ===========================================================================

/// Bytes needed to store `count` indices at `bits` per index.
#[inline]
fn packed_row_bytes(count: usize, bits: u8) -> usize {
    (count * bits as usize + 7) / 8
}

/// Pack a row of u8 indices into a bit-packed byte array.
fn pack_row(indices: &[u8], bits: u8) -> Vec<u8> {
    if bits == 8 {
        return indices.to_vec();
    }
    let n = indices.len();
    let out_len = packed_row_bytes(n, bits);
    let mut out = vec![0u8; out_len];

    match bits {
        1 => {
            for (j, &idx) in indices.iter().enumerate() {
                if idx & 1 != 0 {
                    out[j / 8] |= 1 << (7 - (j % 8));
                }
            }
        }
        2 => {
            for (j, &idx) in indices.iter().enumerate() {
                let byte_pos = j / 4;
                let shift = 6 - 2 * (j % 4);
                out[byte_pos] |= (idx & 0x03) << shift;
            }
        }
        4 => {
            for (j, &idx) in indices.iter().enumerate() {
                let byte_pos = j / 2;
                if j % 2 == 0 {
                    out[byte_pos] |= (idx & 0x0F) << 4;
                } else {
                    out[byte_pos] |= idx & 0x0F;
                }
            }
        }
        _ => {
            // General case for bits 3, 5, 6, 7
            for (j, &idx) in indices.iter().enumerate() {
                let bit_offset = j * bits as usize;
                let byte_idx = bit_offset / 8;
                let bit_pos = bit_offset % 8;
                out[byte_idx] |= (idx as u16 as u8) << bit_pos as u8;
                let overflow = bit_pos + bits as usize;
                if overflow > 8 && byte_idx + 1 < out_len {
                    out[byte_idx + 1] |= (idx >> (8 - bit_pos)) as u8;
                }
            }
        }
    }
    out
}

/// Unpack a single index from a packed byte array.
#[inline(always)]
fn unpack_index(packed: &[u8], j: usize, bits: u8) -> u8 {
    match bits {
        1 => (packed[j / 8] >> (7 - (j % 8))) & 1,
        2 => (packed[j / 4] >> (6 - 2 * (j % 4))) & 0x03,
        4 => {
            let byte = packed[j / 2];
            if j % 2 == 0 { byte >> 4 } else { byte & 0x0F }
        }
        8 => packed[j],
        _ => {
            let bit_offset = j * bits as usize;
            let byte_idx = bit_offset / 8;
            let bit_pos = bit_offset % 8;
            let mask = (1u16 << bits) - 1;
            let mut val = packed[byte_idx] as u16 >> bit_pos;
            if bit_pos + bits as usize > 8 && byte_idx + 1 < packed.len() {
                val |= (packed[byte_idx + 1] as u16) << (8 - bit_pos);
            }
            (val & mask) as u8
        }
    }
}

// ===========================================================================
// CompressedIndex — batch storage + search in the rotated domain
// ===========================================================================

#[wasm_bindgen]
pub struct CompressedIndex {
    dim: usize,
    bits: u8,
    n_vectors: usize,
    /// Bit-packed quantization indices (n_vectors × packed_row_bytes).
    packed: Vec<u8>,
    /// Bytes per packed row (cached for fast offset computation).
    row_bytes: usize,
    /// Original vector norms.
    norms: Vec<f32>,
}

#[wasm_bindgen]
impl CompressedIndex {
    /// Create an empty index ready to receive vectors incrementally.
    ///
    /// - `dim`: embedding dimension (must match the TurboQuantizer)
    /// - `bits`: bits per coordinate (1–8, must match the TurboQuantizer)
    #[wasm_bindgen]
    pub fn new_empty(dim: usize, bits: u8) -> Result<CompressedIndex, JsError> {
        if bits < 1 || bits > 8 {
            return Err(JsError::new("bits must be in [1, 8]"));
        }
        if dim < 2 {
            return Err(JsError::new("dim must be >= 2"));
        }
        let row_bytes = packed_row_bytes(dim, bits);
        Ok(CompressedIndex {
            dim,
            bits,
            n_vectors: 0,
            packed: Vec::new(),
            row_bytes,
            norms: Vec::new(),
        })
    }

    /// Add a single vector to the index (streaming/incremental).
    ///
    /// Algorithm 1 steps:
    ///   1. Extract norm from the embedding
    ///   2. Normalize: x_hat = x / ||x||
    ///   3. Rotate: y = Pi * x_hat
    ///   4. Quantize each coordinate
    ///   5. Pack indices and append to storage
    #[wasm_bindgen]
    pub fn add_vector(
        &mut self,
        quantizer: &TurboQuantizer,
        embedding: &Float32Array,
    ) -> Result<(), JsError> {
        let x = embedding.to_vec();
        if x.len() != self.dim {
            return Err(JsError::new(&format!(
                "Expected dim={}, got {}", self.dim, x.len()
            )));
        }
        if quantizer.dim != self.dim || quantizer.bits != self.bits {
            return Err(JsError::new("Quantizer dim/bits mismatch with index"));
        }

        // 1. Extract norm
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

        // 2-4. Normalize, rotate (SIMD), quantize
        let x_hat: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();
        let y = mat_vec_mul(&quantizer.rotation, &x_hat, self.dim);
        let mut indices = vec![0u8; self.dim];
        for i in 0..self.dim {
            indices[i] = quantize_scalar(y[i], &quantizer.boundaries);
        }

        // 5. Pack and append
        let packed_row = pack_row(&indices, self.bits);
        self.packed.extend_from_slice(&packed_row);
        self.norms.push(norm);
        self.n_vectors += 1;

        Ok(())
    }

    /// Add multiple vectors to the index in batch (appends to existing data).
    ///
    /// `embeddings_flat`: n × dim float32 values, row-major.
    /// `n`: number of vectors to add.
    #[wasm_bindgen]
    pub fn add_vectors(
        &mut self,
        quantizer: &TurboQuantizer,
        embeddings_flat: &Float32Array,
        n: usize,
    ) -> Result<(), JsError> {
        let data = embeddings_flat.to_vec();
        let dim = self.dim;

        if data.len() != n * dim {
            return Err(JsError::new(&format!(
                "Expected {} floats, got {}", n * dim, data.len()
            )));
        }
        if quantizer.dim != self.dim || quantizer.bits != self.bits {
            return Err(JsError::new("Quantizer dim/bits mismatch with index"));
        }

        // Pre-allocate space
        self.packed.reserve(n * self.row_bytes);
        self.norms.reserve(n);

        let mut row_indices = vec![0u8; dim]; // scratch buffer

        for i in 0..n {
            let base = i * dim;
            let x = &data[base..base + dim];

            // Norm
            let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

            // Rotate (SIMD) + quantize
            let x_hat: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();
            let y = mat_vec_mul(&quantizer.rotation, &x_hat, dim);
            for row_idx in 0..dim {
                row_indices[row_idx] = quantize_scalar(y[row_idx], &quantizer.boundaries);
            }

            // Pack and append
            let packed_row = pack_row(&row_indices, self.bits);
            self.packed.extend_from_slice(&packed_row);
            self.norms.push(norm);
            self.n_vectors += 1;
        }

        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn n_vectors(&self) -> usize { self.n_vectors }

    #[wasm_bindgen(getter)]
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.norms.len() * 4
    }

    /// Serialize the index to a compact binary blob (for IndexedDB / localStorage).
    ///
    /// Format: [dim:u32][bits:u8][n_vectors:u32][row_bytes:u32] + packed + norms(f32 LE)
    #[wasm_bindgen]
    pub fn save(&self) -> Uint8Array {
        let header_size = 4 + 1 + 4 + 4; // 13 bytes
        let total = header_size + self.packed.len() + self.norms.len() * 4;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.push(self.bits);
        buf.extend_from_slice(&(self.n_vectors as u32).to_le_bytes());
        buf.extend_from_slice(&(self.row_bytes as u32).to_le_bytes());
        buf.extend_from_slice(&self.packed);
        for &n in &self.norms {
            buf.extend_from_slice(&n.to_le_bytes());
        }

        Uint8Array::from(&buf[..])
    }

    /// Deserialize an index from a binary blob produced by `save()`.
    #[wasm_bindgen]
    pub fn load(data: &Uint8Array) -> Result<CompressedIndex, JsError> {
        let buf = data.to_vec();
        if buf.len() < 13 {
            return Err(JsError::new("Invalid index data: too short"));
        }

        let dim = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let bits = buf[4];
        let n_vectors = u32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]) as usize;
        let row_bytes = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;

        let packed_len = n_vectors * row_bytes;
        let norms_len = n_vectors * 4;
        let expected = 13 + packed_len + norms_len;

        if buf.len() != expected {
            return Err(JsError::new(&format!(
                "Invalid index data: expected {} bytes, got {}", expected, buf.len()
            )));
        }

        let packed = buf[13..13 + packed_len].to_vec();
        let mut norms = Vec::with_capacity(n_vectors);
        let norms_start = 13 + packed_len;
        for i in 0..n_vectors {
            let off = norms_start + i * 4;
            norms.push(f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]));
        }

        Ok(CompressedIndex { dim, bits, n_vectors, packed, row_bytes, norms })
    }

    /// Search for top-k by approximate inner product.
    ///
    /// KEY OPTIMIZATION: We compute scores in the ROTATED domain.
    /// Since Π is orthogonal, ⟨x, q⟩ = ⟨Πx, Πq⟩.
    ///
    /// Steps:
    ///   1. Rotate query once: q_rot = Π · q           — O(d²), done once
    ///   2. For each database vector i:                 — O(d) per vector
    ///      score_i = ‖x_i‖ · Σ_j c_{idx_{i,j}} · q_rot_j
    ///
    /// This avoids the O(d²) inverse rotation per database vector.
    #[wasm_bindgen]
    pub fn search(
        &self,
        quantizer: &TurboQuantizer,
        query: &Float32Array,
        k: usize,
    ) -> Result<js_sys::Uint32Array, JsError> {
        if quantizer.dim != self.dim || quantizer.bits != self.bits {
            return Err(JsError::new("Quantizer dim/bits mismatch with index"));
        }

        let q = query.to_vec();
        if q.len() != self.dim {
            return Err(JsError::new("Query dimension mismatch"));
        }

        // Step 1: Rotate the query — q_rot = Π · q (SIMD-accelerated)
        let q_rot = mat_vec_mul(&quantizer.rotation, &q, self.dim);

        // Step 2: Approximate inner products in rotated domain
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.n_vectors);
        let bits = self.bits;
        let row_bytes = self.row_bytes;
        for i in 0..self.n_vectors {
            let row_start = i * row_bytes;
            let row = &self.packed[row_start..row_start + row_bytes];
            let norm_i = self.norms[i];

            // score = ‖x‖ · Σ_j c_{idx_j} · q_rot_j
            let mut dot = 0.0f32;
            let dim4 = self.dim & !3;
            let mut j = 0usize;
            while j < dim4 {
                let c0 = quantizer.centroids[unpack_index(row, j, bits) as usize];
                let c1 = quantizer.centroids[unpack_index(row, j + 1, bits) as usize];
                let c2 = quantizer.centroids[unpack_index(row, j + 2, bits) as usize];
                let c3 = quantizer.centroids[unpack_index(row, j + 3, bits) as usize];
                dot += c0 * q_rot[j] + c1 * q_rot[j + 1] + c2 * q_rot[j + 2] + c3 * q_rot[j + 3];
                j += 4;
            }
            while j < self.dim {
                let idx = unpack_index(row, j, bits) as usize;
                dot += quantizer.centroids[idx.min(quantizer.centroids.len() - 1)] * q_rot[j];
                j += 1;
            }
            scores.push((i, norm_i * dot));
        }

        // Partial sort: O(n) partition then sort only top-k
        let k_clamped = k.min(scores.len());
        if k_clamped == scores.len() {
            scores.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if k_clamped > 0 {
            scores.select_nth_unstable_by(k_clamped - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scores[..k_clamped].sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let top_k: Vec<u32> = scores.iter().take(k_clamped).map(|(i, _)| *i as u32).collect();

        Ok(js_sys::Uint32Array::from(&top_k[..]))
    }
}

/// Build a compressed index from a flat array of embeddings.
///
/// `embeddings_flat`: n_vectors × dim float32 values, row-major.
/// `n_vectors`: number of vectors.
#[wasm_bindgen]
pub fn build_index(
    quantizer: &TurboQuantizer,
    embeddings_flat: &Float32Array,
    n_vectors: usize,
) -> Result<CompressedIndex, JsError> {
    let data = embeddings_flat.to_vec();
    let dim = quantizer.dim;

    if data.len() != n_vectors * dim {
        return Err(JsError::new(&format!(
            "Expected {} floats, got {}", n_vectors * dim, data.len()
        )));
    }

    let bits = quantizer.bits;
    let row_bytes = packed_row_bytes(dim, bits);
    let mut all_packed = vec![0u8; n_vectors * row_bytes];
    let mut all_norms = Vec::with_capacity(n_vectors);
    let mut row_indices = vec![0u8; dim]; // scratch buffer

    for i in 0..n_vectors {
        let base = i * dim;
        let x = &data[base..base + dim];

        // Norm
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        all_norms.push(norm);
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

        // Rotate (SIMD) + quantize (Algorithm 1, lines 5–6)
        let x_hat: Vec<f32> = x.iter().map(|&v| v * inv_norm).collect();
        let y = mat_vec_mul(&quantizer.rotation, &x_hat, dim);
        for row_idx in 0..dim {
            row_indices[row_idx] = quantize_scalar(y[row_idx], &quantizer.boundaries);
        }

        // Pack this vector's indices
        let packed_row = pack_row(&row_indices, bits);
        all_packed[i * row_bytes..(i + 1) * row_bytes].copy_from_slice(&packed_row);
    }

    Ok(CompressedIndex {
        dim,
        bits,
        n_vectors,
        packed: all_packed,
        row_bytes,
        norms: all_norms,
    })
}
