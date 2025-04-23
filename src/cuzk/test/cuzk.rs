#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use group::{prime::PrimeCurveAffine, Group};
    use halo2curves::bn256::{Fr, G1Affine, G1};

    use crate::cuzk::{
        lib::{sample_points, sample_scalars},
        utils::{to_words_le, to_words_le_from_field},
    };

    use super::*;
    /// Rust implementation of serial transpose algorithm from
    /// https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf.
    /// It simulates running multiple transpositions in parallel, with one thread
    /// per CSR matrix. It does not accept an arbitrary csr_row_ptr array.
    fn calc_start_end(m: u32, n: u32, i: u32) -> (u32, u32) {
        if i < m {
            (i * n, i * n + n)
        } else {
            (m * n, m * n)
        }
    }

    fn get_element<T: Clone + Copy>(arr: &[T], id: i32) -> T {
        if id < 0 {
            arr[arr.len() + id as usize]
        } else {
            arr[id as usize]
        }
    }

    fn update_element<T: Clone + Copy>(arr: &mut Vec<T>, id: i32, val: T) {
        let len = arr.len();
        if id < 0 {
            arr[len + id as usize] = val;
        } else {
            arr[id as usize] = val;
        }
    }

    fn cpu_transpose(
        all_csr_col_idx: Vec<i32>,
        n: u32,
        m: u32,
        num_subtasks: u32,
        input_size: u32,
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut all_csc_col_ptr: Vec<i32> = vec![0; (num_subtasks * (n + 1)) as usize];
        let mut all_csc_row_idx: Vec<i32> = vec![0; (num_subtasks * input_size) as usize];
        let mut all_csc_vals: Vec<i32> = vec![0; (num_subtasks * input_size) as usize];
        let mut all_curr: Vec<i32> = vec![0; (num_subtasks * n) as usize];

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

    fn decompose_scalars_signed<F: PrimeField>(
        scalars: Vec<F>,
        num_words: u32,
        word_size: u32,
    ) -> Vec<Vec<i32>> {
        let l = 1 << word_size;
        let shift = 1 << (word_size - 1);

        let mut as_limbs: Vec<Vec<i32>> = Vec::new();

        for scalar in scalars {
            let limbs = to_words_le_from_field(&scalar, num_words as usize, word_size as usize);
            let mut signed_slices: Vec<i32> = vec![0; limbs.len()];

            let mut carry = 0;
            for i in 0..limbs.len() {
                signed_slices[i] = limbs[i] as i32 + carry;
                if signed_slices[i] >= l / 2 {
                    signed_slices[i] = (l - signed_slices[i]) * -1;
                    if signed_slices[i] == -0 {
                        signed_slices[i] = 0;
                    }
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            if carry == 1 {
                panic!("final carry is 1");
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

    fn cpu_smvp_signed(
        subtask_idx: u32,
        input_size: u32,
        num_columns: u32,
        chunk_size: u32,
        all_csc_col_ptr: Vec<i32>,
        all_csc_val_idxs: Vec<i32>,
        points: Vec<G1Affine>,
    ) -> Vec<G1Affine> {
        let l = 1 << chunk_size;
        let h = l / 2;
        let zero = G1Affine::identity();
        let mut buckets: Vec<G1Affine> = vec![zero; num_columns as usize / 2];

        let rp_offset = subtask_idx * (num_columns + 1);

        for thread_id in 0..num_columns / 2 {
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
                    sum = sum + points[all_csc_val_idxs[subtask_idx * input_size + k]];
                }

                let bucket_idx;
                if h > row_idx {
                    bucket_idx = h - row_idx;
                    sum = -sum;
                } else {
                    bucket_idx = row_idx - h;
                }

                if bucket_idx > 0 {
                    buckets[thread_id as usize] = buckets[thread_id as usize] + sum.into();
                } else {
                    buckets[thread_id as usize] = buckets[thread_id as usize] + zero.into();
                }
            }
        }
        buckets
    }

    fn serial_bucket_reduction(buckets: Vec<G1Affine>) -> G1Affine {
        let mut indices = vec![];
        for i in 1..buckets.len() {
            indices.push(i);
        }
        indices.push(0);

        let mut bucket_sum = G1::ZERO;
        for i in 1..buckets.len() + 1 {
            let b = buckets[indices[i - 1]].multiply(i as u64);
            bucket_sum = bucket_sum + b;
        }
        bucket_sum
    }

    // Perform running sum in the classic fashion - one siumulated thread only
    fn running_sum_bucket_reduction(buckets: Vec<G1Affine>) -> G1Affine {
        let n = buckets.len();
        let mut m = buckets[0];
        let mut g = m;

        for i in 0..n - 1 {
            let idx = n - 1 - i;
            let b = buckets[idx];
            m = m + b;
            g = g + m;
        }
    }

    // Perform running sum with simulated parallelism. It is up to the caller
    // to add the resulting points.

    fn parallel_bucket_reduction(buckets: Vec<G1Affine>, num_threads: u32) -> Vec<G1Affine> {
        let buckets_per_thread = buckets.len() / num_threads;
        let mut bucket_sums: Vec<G1Affine> = vec![];

        for thread_id in 0..num_threads {
            let idx = if thread_id == 0 {
                0
            } else {
                (num_threads - thread_id) * buckets_per_thread
            };

            let m = buckets[idx];
            let mut g = m;

            for i in 0..buckets_per_thread - 1 {
                let idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
                let b = buckets[idx];
                m = m + b;
                g = g + m;
            }

            let s = buckets_per_thread * (num_threads - thread_id - 1);
            if s > 0 {
                g = g + m.multiply(s as u64);
            }

            bucket_sums.push(g);
        }
        bucket_sums
    }

    // The first part of the parallel bucket reduction algo
    fn parallel_bucket_reduction_1(
        buckets: Vec<G1Affine>,
        num_threads: u32,
    ) -> (Vec<G1Affine>, Vec<G1Affine>) {
        let buckets_per_thread = buckets.len() / num_threads;
        let mut g_points: Vec<G1Affine> = vec![];
        let mut m_points: Vec<G1Affine> = vec![];

        for thread_id in 0..num_threads {
            let idx = if thread_id == 0 {
                0
            } else {
                (num_threads - thread_id) * buckets_per_thread
            };

            let m = buckets[idx];
            let mut g = m;

            for i in 0..buckets_per_thread - 1 {
                let idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
                let b = buckets[idx];
                m = m + b;
                g = g + m;
            }

            g_points.push(g);
            m_points.push(m);
        }
        (g_points, m_points)
    }

    // The second part of the parallel bucket reduction algo

    fn parallel_bucket_reduction_2(
        g_points: Vec<G1Affine>,
        m_points: Vec<G1Affine>,
        num_buckets: u32,
        num_threads: u32,
    ) -> Vec<G1Affine> {
        let buckets_per_thread = num_buckets / num_threads;
        let mut result: Vec<G1Affine> = vec![];

        for thread_id in 0..num_threads {
            let g = g_points[thread_id];
            let m = m_points[thread_id];
            let s = buckets_per_thread * (num_threads - thread_id - 1);
            if s > 0 {
                g = g + m.multiply(s as u64);
            }
            result.push(g);
        }
        result
    }

    #[test]
    fn test_cuzk() {
        let input_size = 16;
        let chunk_size = 4;
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;

        let scalars = sample_scalars::<Fr>(input_size);
        let points = sample_points::<G1Affine>(input_size);

        let decomposed_scalars = decompose_scalars_signed(scalars, num_subtasks, chunk_size);

        let bucket_sums = vec![];
        // Perform multiple transpositions "in parallel"}
        let (all_csc_col_ptr, all_csc_row_idx, all_csc_vals) = cpu_transpose(
            decomposed_scalars.concat(),
            num_columns,
            num_rows,
            num_subtasks,
            input_size,
        );

        for subtask_idx in 0..num_subtasks {
            // Perform SMVP
            let buckets = cpu_smvp_signed(
                subtask_idx,
                input_size,
                num_columns,
                chunk_size,
                all_csc_col_ptr,
                all_csc_vals,
                points,
            );

            let buckets_sum_serial = serial_bucket_reduction(buckets);
            let buckets_sum_rs = running_sum_bucket_reduction(buckets);

            // Use the full pBucketPointReduciton algo
            let bucket_sum = G1::ZERO;
            for b in parallel_bucket_reduction(buckets) {
                bucket_sum = bucket_sum + b;
            }

            assert_eq!(bucket_sum_serial, bucket_sum);
            assert_eq!(bucket_sum_rs, bucket_sum);

            bucket_sums.push(bucket_sum);

            let num_buckets = buckets.len();
            let (g_points, m_points) = parallel_bucket_reduction_1(buckets);

            let p_result =
                parallel_bucket_reduction_2(g_points, m_points, num_buckets, num_subtasks);

            let bucket_sum_2 = G1::ZERO;
            for b in p_result {
                bucket_sum_2 = bucket_sum_2 + b;
            }

            assert_eq!(bucket_sum_serial, bucket_sum_2);
            assert_eq!(bucket_sum_rs, bucket_sum_2);

            // Horner's rule

            let m = 1 << chunk_size;
            let result = bucket_sums[bucket_sums.len() - 1];
            for i in (0..bucket_sums.len() - 1).rev() {
                result = result.multiply(m);
                result = result.add(bucket_sums[i]);
            }

            let result_affine = result.toAffine();

            let expected = G1::ZERO;
            for i in 0..input_size {
                if scalars[i] != G1::ZERO {
                    let p = points[i].multiply(scalars[i]);
                    expected = expected.add(p);
                }
            }
            let expected_affine = expected.toAffine();

            assert_eq!(result_affine.x, expected_affine.x);
            assert_eq!(result_affine.y, expected_affine.y);
        }
    }
}
