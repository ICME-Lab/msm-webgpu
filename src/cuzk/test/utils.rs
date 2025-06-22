use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use halo2curves::CurveAffine;

use crate::cuzk::utils::to_words_le_from_field;

/// Rust implementation of serial transpose algorithm from
/// https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf.
/// It simulates running multiple transpositions in parallel, with one thread
/// per CSR matrix. It does not accept an arbitrary csr_row_ptr array.
pub fn calc_start_end(m: usize, n: usize, i: usize) -> (usize, usize) {
    if i < m {
        (i * n, i * n + n)
    } else {
        (m * n, m * n)
    }
}

pub fn get_element(arr: &[i32], id: i32) -> i32 {
    if id < 0 {
        if (arr.len() as i32 + id) < 0 {
            return 0;
        }
        arr[arr.len() + id as usize]
    } else {
        if id >= arr.len() as i32 {
            return 0;
        }
        arr[id as usize]
    }
}

pub fn get_point_element<C: CurveAffine>(arr: &[C], id: i32) -> C {
    if id < 0 {
        if (arr.len() as i32 + id) < 0 {
            return C::identity();
        }
        arr[arr.len() + id as usize]
    } else {
        if id >= arr.len() as i32 {
            return C::identity();
        }
        arr[id as usize]
    }
}

/// Update element
pub fn update_element(arr: &mut [i32], id: i32, val: i32) {
    let len = arr.len();
    if id < 0 {
        arr[len + id as usize] = val;
    } else {
        if id >= arr.len() as i32 {
            return;
        }
        arr[id as usize] = val;
    }
}

/// CPU transpose
pub fn cpu_transpose(
    all_csr_col_idx: Vec<i32>,
    n: usize,
    m: usize,
    num_subtasks: usize,
    input_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut all_csc_col_ptr: Vec<i32> = vec![0; num_subtasks * (n + 1)];
    let mut all_csc_row_idx: Vec<i32> = vec![0; num_subtasks * input_size];
    let mut all_csc_vals: Vec<i32> = vec![0; num_subtasks * input_size];
    let mut all_curr: Vec<i32> = vec![0; num_subtasks * n];

    for subtask_idx in 0..num_subtasks {
        let ccp_offset = (subtask_idx * (n + 1)) as i32;
        let cci_offset = (subtask_idx * input_size) as i32;
        let curr_offset = (subtask_idx * n) as i32;

        for i in 0..m {
            let (start, end) = calc_start_end(m, n, i);
            for j in start..end {
                let tmp = get_element(&all_csr_col_idx, cci_offset + j as i32);
                let idx = ccp_offset + tmp + 1;
                let val = get_element(&all_csc_col_ptr, idx);
                update_element(&mut all_csc_col_ptr, idx, val + 1);
            }
        }

        for i in 0..n {
            let idx = ccp_offset + i as i32 + 1;
            let val = get_element(&all_csc_col_ptr, idx);
            let prev_val = get_element(&all_csc_col_ptr, ccp_offset + i as i32);
            update_element(&mut all_csc_col_ptr, idx, val + prev_val);
        }

        let mut val = 0;
        for i in 0..m {
            let (start, end) = calc_start_end(m, n, i);
            for j in start..end {
                let tmp = get_element(&all_csr_col_idx, cci_offset + j as i32);
                let idx = curr_offset + tmp;
                let col = ccp_offset + tmp;

                let loc = get_element(&all_csc_col_ptr, col) + get_element(&all_curr, idx);
                let curr_val = get_element(&all_curr, idx);
                update_element(&mut all_curr, idx, curr_val + 1);
                update_element(
                    &mut all_csc_row_idx,
                    subtask_idx as i32 * input_size as i32 + loc,
                    i as i32,
                );
                update_element(&mut all_csc_vals, cci_offset + loc, val);

                val += 1;
            }
        }
    }
    (all_csc_col_ptr, all_csc_row_idx, all_csc_vals)
}

/// Decompose scalars signed
pub fn decompose_scalars_signed<F: PrimeField>(
    scalars: &[F],
    num_words: usize,
    word_size: usize,
) -> Vec<Vec<i32>> {
    let l = 1 << word_size;
    let shift = 1 << (word_size - 1);

    let mut as_limbs: Vec<Vec<i32>> = Vec::new();

    for scalar in scalars {
        let limbs = to_words_le_from_field(scalar, num_words, word_size);
        let mut signed_slices: Vec<i32> = vec![0; limbs.len()];

        let mut carry = 0;
        for i in 0..limbs.len() {
            signed_slices[i] = limbs[i] as i32 + carry;
            if signed_slices[i] >= l / 2 {
                signed_slices[i] = -(l - signed_slices[i]);
                // if signed_slices[i] == 0 {
                //     signed_slices[i] = 0;
                // }
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if carry == 1 {
            // TODO: Review this
            // panic!("final carry is 1");
            println!("Carrying 1");
            println!("Scalar: {:?}", scalar);
            println!("Limbs: {:?}", limbs);
            signed_slices.push(carry);
        }
        as_limbs.push(signed_slices.iter().map(|x| x + shift).collect());
    }
    let mut result: Vec<Vec<i32>> = Vec::new();
    for i in 0..num_words {
        let t = as_limbs.iter().map(|limbs| limbs[i]).collect();
        result.push(t);
    }
    result
}

/**
 * Perform SMVP with signed bucket indices
 */
pub fn cpu_smvp_signed<C: CurveAffine>(
    subtask_idx: usize,
    input_size: usize,
    num_columns: usize,
    chunk_size: usize,
    all_csc_col_ptr: &[i32],
    all_csc_val_idxs: &[i32],
    points: &[C],
) -> Vec<C::Curve> {
    let l = 1 << chunk_size;
    let h = l / 2;
    let zero = C::Curve::identity();
    let mut buckets: Vec<C::Curve> = vec![zero; num_columns / 2];

    let rp_offset = subtask_idx * (num_columns + 1);

    for (thread_id, bucket) in buckets.iter_mut().enumerate() {
        for j in 0..2 {
            let mut row_idx = thread_id + num_columns / 2;
            if j == 1 {
                row_idx = num_columns / 2 - thread_id;
            }
            if thread_id == 0 && j == 0 {
                row_idx = 0;
            }

            let row_begin = all_csc_col_ptr[rp_offset + row_idx];
            let row_end = all_csc_col_ptr[rp_offset + row_idx + 1];

            let mut sum = zero;
            for k in row_begin..row_end {
                let idx = subtask_idx as i32 * input_size as i32 + k;
                let val = get_element(all_csc_val_idxs, idx);
                let point = get_point_element(points, val);
                sum += C::Curve::from(point);
            }

            let bucket_idx;
            if h > row_idx {
                bucket_idx = h - row_idx;
                sum = -sum;
            } else {
                bucket_idx = row_idx - h;
            }

            if bucket_idx > 0 {
                *bucket += sum;
            } else {
                *bucket += zero;
            }
        }
    }
    buckets
}

/// Serial bucket reduction
pub fn serial_bucket_reduction<C: CurveAffine>(buckets: &[C::Curve]) -> C::Curve {
    let mut indices = vec![];
    for i in 1..buckets.len() {
        indices.push(i);
    }
    indices.push(0);

    let mut bucket_sum = C::Curve::identity();
    for i in 1..buckets.len() + 1 {
        let b = buckets[indices[i - 1]] * C::Scalar::from(i as u64);
        bucket_sum += b;
    }
    bucket_sum
}

/// Perform running sum in the classic fashion - one siumulated thread only
pub fn running_sum_bucket_reduction<C: CurveAffine>(buckets: &[C::Curve]) -> C::Curve {
    let n = buckets.len();
    let mut m = buckets[0];
    let mut g = m;

    for i in 0..n - 1 {
        let idx = n - 1 - i;
        let b = buckets[idx];
        m += b;
        g += m;
    }

    g
}

/// Perform running sum with simulated parallelism. It is up to the caller
/// to add the resulting points.
pub fn parallel_bucket_reduction<C: CurveAffine>(buckets: &[C::Curve], num_threads: usize) -> Vec<C::Curve> {
    let buckets_per_thread = buckets.len() / num_threads;
    let mut bucket_sums: Vec<C::Curve> = vec![];

    for thread_id in 0..num_threads {
        let idx = if thread_id == 0 {
            0
        } else {
            (num_threads - thread_id) * buckets_per_thread
        };

        let mut m = buckets[idx];
        let mut g = m;

        for i in 0..(buckets_per_thread - 1) {
            let idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
            let b = buckets[idx];
            m += b;
            g += m;
        }

        let s = buckets_per_thread * (num_threads - thread_id - 1);
        if s > 0 {
            g += m * C::Scalar::from(s as u64);
        }

        bucket_sums.push(g);
    }
    bucket_sums
}

/// The first part of the parallel bucket reduction algo
pub fn parallel_bucket_reduction_1<C: CurveAffine>(
    buckets: &[C::Curve],
    num_threads: usize,
) -> (Vec<C::Curve>, Vec<C::Curve>) {
    let buckets_per_thread = buckets.len() / num_threads;
    let mut g_points: Vec<C::Curve> = vec![];
    let mut m_points: Vec<C::Curve> = vec![];

    for thread_id in 0..num_threads {
        let idx = if thread_id == 0 {
            0
        } else {
            (num_threads - thread_id) * buckets_per_thread
        };

        let mut m = buckets[idx];
        let mut g = m;

        for i in 0..(buckets_per_thread - 1) {
            let idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
            let b = buckets[idx];
            m += b;
            g += m;
        }

        g_points.push(g);
        m_points.push(m);
    }
    (g_points, m_points)
}

/// The second part of the parallel bucket reduction algo
pub fn parallel_bucket_reduction_2<C: CurveAffine>(
    g_points: Vec<C::Curve>,
    m_points: Vec<C::Curve>,
    num_buckets: usize,
    num_threads: usize,
) -> Vec<C::Curve> {
    let buckets_per_thread = num_buckets / num_threads;
    let mut result: Vec<C::Curve> = vec![];

    for thread_id in 0..num_threads {
        let mut g = g_points[thread_id];
        let m = m_points[thread_id];
        let s = buckets_per_thread * (num_threads - thread_id - 1);
        if s > 0 {
            g += m * C::Scalar::from(s as u64);
        }
        result.push(g);
    }
    result
}