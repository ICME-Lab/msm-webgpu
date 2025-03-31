use halo2curves::bn256::{G1Affine, G1, Fr};
use group::{Curve, Group};
use ff::{Field, PrimeField, PrimeFieldBits};

use crate::halo2curves::utils::{field_to_bytes, field_to_u16_vec};

const POINTS_PER_INVOCATION: usize = 64;

// n - number of points/scalars
// b - bit length of the scalar
const B: usize = 256;
// c - window chunk size in bits
const C: usize = 8;
// w - number of windows per scalar (b/c)
const W: usize = B / C; // 32


const BUCKETS_PER_WINDOW: usize = 1 << C; // 8 bits per window
const TOTAL_BUCKETS: usize = W * BUCKETS_PER_WINDOW;
// i - a point/scalar index, 1...n
// j - a window index, 0...W-1
// k - a bucket index, 0...2^C - 1
// s_i[j] - value of the j-th chunk of the i-th scalar
// B[j, k] - accumulator bucket


pub fn emulate_pippenger(points: &[G1Affine], scalars: &[Fr], gidx: usize) -> G1 {
    let base = gidx * POINTS_PER_INVOCATION;
    let points = &points[base..base + POINTS_PER_INVOCATION];
    let scalars = &scalars[base..base + POINTS_PER_INVOCATION];
    let scalars_and_points = scalars.iter().zip(points.iter()).collect::<Vec<_>>();

    let mut buckets = vec![G1::identity(); TOTAL_BUCKETS];
    let mut windows = vec![G1::identity(); W];

    // Bucket accumulation
    for (scalar, point) in scalars_and_points {
        for j in 0..W {
            let u8_scalar = field_to_bytes(*scalar);
            let s_j = u8_scalar[j];
            if s_j != 0 {
                buckets[j * BUCKETS_PER_WINDOW + s_j as usize] += point;
            }
        }
    }

    // Bucket reduction
    for j in 0..W {
        let mut sum = G1::identity();
        let mut sum_of_sums = G1::identity();
        for k in (1..BUCKETS_PER_WINDOW).rev() {
            sum += buckets[j * BUCKETS_PER_WINDOW + k];
            sum_of_sums += sum;
        }
        windows[j] = sum_of_sums;
    }


    // Final reduction
    let mut result = G1::identity();
    for j in (0..W).rev() {
        result = windows[j] + result * Fr::from(2u64.pow(C as u32));
    }

    result
}

pub fn emulate_pippenger_gpu(points: &[G1Affine], scalars: &[Fr]) -> G1 {
    assert_eq!(points.len(), scalars.len());
    assert_eq!(points.len() % POINTS_PER_INVOCATION, 0);

    let num_invocations = points.len() / POINTS_PER_INVOCATION;
    println!("num_invocations: {:?}", num_invocations);
    let mut partial_results = vec![G1::identity(); num_invocations];

    // === Simulate: @compute @workgroup_size(1) main() ===
    for gidx in 0..num_invocations {
        partial_results[gidx] = emulate_pippenger(points, scalars, gidx);
    }


    // === Simulate: @compute @workgroup_size(256) aggregate() ===
    let mut reduced = partial_results.clone();
    let split = num_invocations / 256;

    // Step 1: Each thread (idx ∈ 0..255) reduces its vertical slice
    for lidx in 0..256 {
        for j in 0..split {
            reduced[lidx] += partial_results[lidx + j * 256];
        }
    }

    // Step 2: Binary tree reduction across threads
    while reduced.len() > 1 {
        let half = reduced.len() / 2;
        for i in 0..half {
            reduced[i] = reduced[i] + reduced[i + half];
        }
        reduced.truncate(half);
    }

    reduced[0] // Final result
}
