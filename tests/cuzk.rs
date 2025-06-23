
#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fr, G1Affine, G1};
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Scalar as PallasScalar, Point as PallasPoint};
    use halo2curves::secp256k1::{Secp256k1Affine, Fq as Secp256k1Fq, Secp256k1};
    use halo2curves::secq256k1::Secq256k1Affine;
    use halo2curves::CurveAffine;
    use msm_webgpu::cuzk::test::utils::*;
    use msm_webgpu::{cpu_msm, sample_points, sample_scalars};
    use group::{Curve, Group};

    fn test_cuzk<C: CurveAffine>() {

        let input_size: usize = 1 << 8;
        let next_power_of_two = input_size.next_power_of_two();
        let scalars = sample_scalars::<C::Scalar>(input_size);
        let points = sample_points::<C>(input_size);

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

            let buckets_sum_serial = serial_bucket_reduction::<C>(&buckets);
            let buckets_sum_rs = running_sum_bucket_reduction::<C>(&buckets);

            let mut bucket_sum = C::Curve::identity();
            for b in parallel_bucket_reduction::<C>(&buckets, 4) {
                bucket_sum = bucket_sum + b;
            }

            assert_eq!(buckets_sum_serial, bucket_sum);
            assert_eq!(buckets_sum_rs, bucket_sum);

            bucket_sums.push(bucket_sum);

            let num_buckets = buckets.len();
            let (g_points, m_points) = parallel_bucket_reduction_1::<C>(&buckets, 4);

            let p_result = parallel_bucket_reduction_2::<C>(g_points, m_points, num_buckets, 4);

            let mut bucket_sum_2 = C::Curve::identity();
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
            result = result * C::Scalar::from(m as u64);
            result = result + bucket_sums[i];
        }

        let result_affine = result.to_affine();

        let expected = cpu_msm(&points, &scalars);
        let expected_affine = expected.to_affine();

        assert_eq!(result_affine, expected_affine);
    }


    #[test]
    fn test_cuzk_bn256() {
        test_cuzk::<G1Affine>();
    }

    #[test]
    fn test_cuzk_pallas() {
       test_cuzk::<PallasAffine>(); 
    }

    #[test]
    fn test_cuzk_secp256k1() {
        test_cuzk::<Secp256k1Affine>(); 
    }

    #[test]
    fn test_cuzk_secq256k1() {
        test_cuzk::<Secq256k1Affine>(); 
    }

}
