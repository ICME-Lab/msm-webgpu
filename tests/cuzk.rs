
#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fr, G1Affine, G1};
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Scalar as PallasScalar, Point as PallasPoint};
    use halo2curves::secp256k1::{Secp256k1Affine, Fq as Secp256k1Fq, Secp256k1};
    use msm_webgpu::cuzk::test::utils::*;
    use msm_webgpu::{cpu_msm, sample_points, sample_scalars};
    use group::{Curve, Group};

    #[test]
    fn test_cuzk_bn256() {

        let input_size: usize = 1 << 16;
        let next_power_of_two = input_size.next_power_of_two();
        let scalars = sample_scalars::<Fr>(input_size);
        let points = sample_points::<G1Affine>(input_size);

        let input_size = next_power_of_two;

        let chunk_size = if input_size >= 65536 { 16 } else { 4 };
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;


        let decomposed_scalars = decompose_scalars_signed(&scalars, num_subtasks, chunk_size);

        let mut bucket_sums = vec![];
        // Perform multiple transpositions "in parallel"}
        let (all_csc_col_ptr, _, all_csc_vals) = cpu_transpose(
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
                &all_csc_col_ptr,
                &all_csc_vals,
                &points,
            );

            let buckets_sum_serial = serial_bucket_reduction::<G1Affine>(&buckets);
            let buckets_sum_rs = running_sum_bucket_reduction::<G1Affine>(&buckets);

            let mut bucket_sum = G1::identity();
            for b in parallel_bucket_reduction::<G1Affine>(&buckets, 4) {
                bucket_sum = bucket_sum + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum);
            assert_eq!(buckets_sum_rs, bucket_sum);

            bucket_sums.push(bucket_sum);

            let num_buckets = buckets.len();
            let (g_points, m_points) = parallel_bucket_reduction_1::<G1Affine>(&buckets, 4);

            let p_result = parallel_bucket_reduction_2::<G1Affine>(g_points, m_points, num_buckets, 4);

            let mut bucket_sum_2 = G1::identity();
            for b in p_result {
                bucket_sum_2 = bucket_sum_2 + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum_2);
            assert_eq!(buckets_sum_rs, bucket_sum_2);
        }

        // Horner's rule

        let m = 1 << chunk_size;
        let mut result = bucket_sums[bucket_sums.len() - 1];
        for i in (0..bucket_sums.len() - 1).rev() {
            result = result * Fr::from(m as u64);
            result = result + bucket_sums[i];
        }

        let result_affine = result.to_affine();

        let expected = cpu_msm(&points, &scalars);
        let expected_affine = expected.to_affine();

        assert_eq!(result_affine.x, expected_affine.x);
        assert_eq!(result_affine.y, expected_affine.y);
    }


    #[test]
    fn test_cuzk_pallas() {

        let input_size: usize = 1 << 16;
        let next_power_of_two = input_size.next_power_of_two();
        let scalars = sample_scalars::<PallasScalar>(input_size);
        let points = sample_points::<PallasAffine>(input_size);

        let input_size = next_power_of_two;

        let chunk_size = if input_size >= 65536 { 16 } else { 4 };
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;


        let decomposed_scalars = decompose_scalars_signed(&scalars, num_subtasks, chunk_size);

        let mut bucket_sums = vec![];
        // Perform multiple transpositions "in parallel"}
        let (all_csc_col_ptr, _, all_csc_vals) = cpu_transpose(
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
                &all_csc_col_ptr,
                &all_csc_vals,
                &points,
            );

            let buckets_sum_serial = serial_bucket_reduction::<PallasAffine>(&buckets);
            let buckets_sum_rs = running_sum_bucket_reduction::<PallasAffine>(&buckets);

            let mut bucket_sum = PallasPoint::identity();
            for b in parallel_bucket_reduction::<PallasAffine>(&buckets, 4) {
                bucket_sum = bucket_sum + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum);
            assert_eq!(buckets_sum_rs, bucket_sum);

            bucket_sums.push(bucket_sum);

            let num_buckets = buckets.len();
            let (g_points, m_points) = parallel_bucket_reduction_1::<PallasAffine>(&buckets, 4);

            let p_result = parallel_bucket_reduction_2::<PallasAffine>(g_points, m_points, num_buckets, 4);

            let mut bucket_sum_2 = PallasPoint::identity();
            for b in p_result {
                bucket_sum_2 = bucket_sum_2 + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum_2);
            assert_eq!(buckets_sum_rs, bucket_sum_2);
        }

        // Horner's rule

        let m = 1 << chunk_size;
        let mut result = bucket_sums[bucket_sums.len() - 1];
        for i in (0..bucket_sums.len() - 1).rev() {
            result = result * PallasScalar::from(m as u64);
            result = result + bucket_sums[i];
        }

        let result_affine = result.to_affine();

        let expected = cpu_msm(&points, &scalars);
        let expected_affine = expected.to_affine();

        assert_eq!(result_affine, expected_affine);
    }

    #[test]
    fn test_cuzk_secp256k1() {

        let input_size: usize = 1 << 16;
        let next_power_of_two = input_size.next_power_of_two();
        let scalars = sample_scalars::<Secp256k1Fq>(input_size);
        let points = sample_points::<Secp256k1Affine>(input_size);

        let input_size = next_power_of_two;

        let chunk_size = if input_size >= 65536 { 16 } else { 4 };
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;


        let decomposed_scalars = decompose_scalars_signed(&scalars, num_subtasks, chunk_size);

        let mut bucket_sums = vec![];
        // Perform multiple transpositions "in parallel"}
        let (all_csc_col_ptr, _, all_csc_vals) = cpu_transpose(
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
                &all_csc_col_ptr,
                &all_csc_vals,
                &points,
            );

            let buckets_sum_serial = serial_bucket_reduction::<Secp256k1Affine>(&buckets);
            let buckets_sum_rs = running_sum_bucket_reduction::<Secp256k1Affine>(&buckets);

            let mut bucket_sum = Secp256k1::identity();
            for b in parallel_bucket_reduction::<Secp256k1Affine>(&buckets, 4) {
                bucket_sum = bucket_sum + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum);
            assert_eq!(buckets_sum_rs, bucket_sum);

            bucket_sums.push(bucket_sum);

            let num_buckets = buckets.len();
            let (g_points, m_points) = parallel_bucket_reduction_1::<Secp256k1Affine>(&buckets, 4);

            let p_result = parallel_bucket_reduction_2::<Secp256k1Affine>(g_points, m_points, num_buckets, 4);

            let mut bucket_sum_2 = Secp256k1::identity();
            for b in p_result {
                bucket_sum_2 = bucket_sum_2 + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum_2);
            assert_eq!(buckets_sum_rs, bucket_sum_2);
        }

        // Horner's rule

        let m = 1 << chunk_size;
        let mut result = bucket_sums[bucket_sums.len() - 1];
        for i in (0..bucket_sums.len() - 1).rev() {
            result = result * Secp256k1Fq::from(m as u64);
            result = result + bucket_sums[i];
        }

        let result_affine = result.to_affine();

        let expected = cpu_msm(&points, &scalars);
        let expected_affine = expected.to_affine();

        assert_eq!(result_affine.x, expected_affine.x);
        assert_eq!(result_affine.y, expected_affine.y);
    }

}
