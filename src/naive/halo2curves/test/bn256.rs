#[cfg(test)]
mod tests {
    use std::time::Instant;

    use group::cofactor::CofactorCurveAffine;
    use rand::Rng;

    use crate::naive::gpu::test::pippenger::{
        emulate_bucket_accumulation, emulate_bucket_reduction, emulate_pippenger_gpu,
    };

    use crate::naive::halo2curves::{cpu_msm, fields_to_bytes_montgomery, points_to_bytes, run_webgpu_msm, sample_points, sample_scalars, scalars_to_bytes};
    use crate::naive::halo2curves::utils::{
        cast_u8_to_u16, cast_u8_to_u32, fields_to_u32_vec, u16_vec_to_fields, u32_vec_to_fields,
    };
    use crate::naive::utils::files::{
        load_bn254_constants_shader_code, load_bn254_field_shader_code,
        load_bn254_field_to_bytes_shader_code, load_bn254_pippenger_phases_shader_code,
        load_bn254_point_msm_shader_code, load_bn254_point_shader_code,
        load_bn254_sum_of_sums_shader_code,
    };
    use crate::naive::utils::montgomery::{field_to_bytes_montgomery, u16_vec_to_fields_montgomery};

    use crate::naive::{gpu, halo2curves::utils::fields_to_u16_vec};
    use ff::{Field, PrimeField};
    use group::{Curve, Group};
    use halo2curves::{
        bn256::{Fq, Fr, G1Affine, G1},
        CurveExt,
    };
    use rand::thread_rng;

    #[test]
    fn test_bn256() {
        let sample_size = 80000;
        let scalars = sample_scalars::<Fr>(sample_size);
        let points = sample_points::<G1Affine>(sample_size);

        let now = Instant::now();
        let fast = cpu_msm(&points, &scalars);
        println!("Fast Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now();
        let webgpu = run_webgpu_msm(&points, &scalars);
        println!("WebGPU Elapsed: {:.2?}", now.elapsed());
        assert_eq!(fast, webgpu);
    }

    #[test]
    fn test_pippenger_emul() {
        let sample_size = 100000;
        let scalars = sample_scalars::<Fr>(sample_size);
        let points = sample_points::<G1Affine>(sample_size);
        let now = Instant::now();
        let fast = cpu_msm(&points, &scalars);
        println!("Fast Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now();
        let result = emulate_pippenger_gpu(&points, &scalars);
        println!("WebGPU Elapsed: {:.2?}", now.elapsed());

        assert_eq!(result, fast);
    }

    #[test]
    fn test_fields_to_u16_vec() {
        let fields = sample_scalars::<Fq>(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_fields_to_u32_vec() {
        let fields = sample_scalars::<Fq>(10);
        let u32_vec = fields_to_u32_vec(&fields);
        let new_fields = u32_vec_to_fields(&u32_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_u16_field_casting() {
        let fields = sample_scalars::<Fq>(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let u16_vec_as_u8 = u16_vec
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        let casted_16 = cast_u8_to_u16(&u16_vec_as_u8);
        let new_fields = u16_vec_to_fields(&casted_16);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_u32_field_casting() {
        let fields = sample_scalars::<Fq>(10);
        let u32_vec = fields_to_u32_vec(&fields);
        let u32_vec_as_u8 = u32_vec
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        let casted_32 = cast_u8_to_u32(&u32_vec_as_u8);
        let new_fields = u32_vec_to_fields(&casted_32);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_field_to_bytes() {
        let field = Fq::random(&mut thread_rng());
        let bytes = fields_to_bytes_montgomery(&vec![field]);
        let shader_code = load_bn254_field_to_bytes_shader_code();
        let gpu_bytes = pollster::block_on(gpu::test::ops::field_to_bytes(&shader_code, &bytes));
        let bytes = field_to_bytes_montgomery(&field);
        let gpu_bytes = cast_u8_to_u32(&gpu_bytes)
            .iter()
            .map(|x| *x as u8)
            .collect::<Vec<_>>();
        assert_eq!(bytes, gpu_bytes);
    }

    #[test]
    fn test_constants() {
        let random_size = thread_rng().gen_range(1..1000);
        let fields = sample_scalars::<Fq>(random_size);
        let bytes = scalars_to_bytes(&fields);
        let shader_code = load_bn254_constants_shader_code();
        let result = pollster::block_on(gpu::test::constants::run_constants(&shader_code, &bytes));
        println!("Result: {:?}", result);
        assert_eq!(result[0], fields.len() as u32);
        let num_invocations = (fields.len() + 64 - 1) / 64;
        assert_eq!(result[1], num_invocations as u32);
    }

    #[test]
    fn test_bucket_accumulation() {
        let sample_size = 64;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);
        let points_bytes = points_to_bytes(&points);
        let scalars_bytes = scalars_to_bytes(&scalars);
        let shader_code = load_bn254_pippenger_phases_shader_code();
        let result = pollster::block_on(gpu::test::pippenger_phases::run_bucket_accumulation(
            &shader_code,
            &points_bytes,
            &scalars_bytes,
        ));
        let gpu_buckets_in_fields: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let gpu_buckets = gpu_buckets_in_fields
            .chunks_exact(3)
            .map(|x| G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap())
            .collect::<Vec<_>>();
        let num_invocations = (points.len() + 64 - 1) / 64;
        let total_buckets = 8192;
        let mut emulated_buckets = vec![G1::identity(); total_buckets * num_invocations];
        for i in 0..num_invocations {
            emulate_bucket_accumulation(&points, &scalars, &mut emulated_buckets, i);
        }
        for (i, (gpu_bucket, emulated_bucket)) in
            gpu_buckets.iter().zip(emulated_buckets.iter()).enumerate()
        {
            if gpu_bucket.is_identity().unwrap_u8() == 0
                || emulated_bucket.is_identity().unwrap_u8() == 0
            {
                println!("Index: {:?}", i);
                println!("GPU Bucket: {:?}", gpu_bucket);
                println!("Emulated Bucket: {:?}", emulated_bucket);
                assert_eq!(gpu_bucket, emulated_bucket);
            }
        }
    }

    #[test]
    fn test_bucket_reduction() {
        let sample_size = 64;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);
        let points_bytes = points_to_bytes(&points);
        let scalars_bytes = scalars_to_bytes(&scalars);

        let num_invocations = (points.len() + 64 - 1) / 64;
        let total_buckets = 8192;
        let total_windows = 32;
        let mut emulated_buckets = vec![G1::identity(); total_buckets * num_invocations];
        let mut emulated_windows = vec![G1::identity(); num_invocations * total_windows];
        for i in 0..num_invocations {
            emulate_bucket_accumulation(&points, &scalars, &mut emulated_buckets, i);
            emulate_bucket_reduction(&mut emulated_buckets, &mut emulated_windows, 0);
        }
        let shader_code = load_bn254_pippenger_phases_shader_code();
        let result = pollster::block_on(gpu::test::pippenger_phases::run_bucket_reduction(
            &shader_code,
            &points_bytes,
            &scalars_bytes,
        ));
        let gpu_windows_in_fields: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let gpu_windows = gpu_windows_in_fields
            .chunks_exact(3)
            .enumerate()
            .map(|(i, x)| {
                let p = G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap();
                if p.is_identity().unwrap_u8() == 0 {
                    println!("Index: {:?}", i);
                    println!("GPU Window: {:?}", p);
                }
                p
            })
            .collect::<Vec<_>>();
        for (i, (gpu_window, emulated_window)) in
            gpu_windows.iter().zip(emulated_windows.iter()).enumerate()
        {
            if gpu_window.is_identity().unwrap_u8() == 0
                || emulated_window.is_identity().unwrap_u8() == 0
            {
                println!("Index: {:?}", i);
                println!("GPU Window: {:?}", gpu_window);
                println!("Emulated Window: {:?}", emulated_window);
                assert_eq!(gpu_window, emulated_window);
            }
        }
    }

    #[test]
    fn test_field_mul() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a * b;

        let a_bytes = fields_to_bytes_montgomery(&vec![a]);
        let b_bytes = fields_to_bytes_montgomery(&vec![b]);

        let shader_code = load_bn254_field_shader_code();

        let result =
            pollster::block_on(gpu::test::ops::field_mul(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_add() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a + b;

        let a_bytes = fields_to_bytes_montgomery(&vec![a]);
        let b_bytes = fields_to_bytes_montgomery(&vec![b]);

        let shader_code = load_bn254_field_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::field_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_sub() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a - b;

        let a_bytes = fields_to_bytes_montgomery(&vec![a]);
        let b_bytes = fields_to_bytes_montgomery(&vec![b]);

        let shader_code = load_bn254_field_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::field_sub(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_point_add() {
        let a = G1Affine::random(&mut thread_rng());
        let b = G1Affine::random(&mut thread_rng());
        let c = a + b;

        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_bn254_point_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_add_identity_right() {
        let a = G1Affine::random(&mut thread_rng());
        let b = G1Affine::identity();

        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_bn254_point_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        println!("GPU Result: {:?}", gpu_result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result, G1::from(a));
    }

    #[test]
    fn test_point_add_identity_left() {
        let a = G1Affine::identity();
        let b = G1Affine::random(&mut thread_rng());

        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_bn254_point_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result, G1::from(b));
    }

    #[test]
    fn test_point_double() {
        let a = G1Affine::random(&mut thread_rng());
        let c = a + a;

        let a_bytes = points_to_bytes(&vec![a]);

        let shader_code = load_bn254_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_double(&shader_code, &a_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_identity() {
        let a = G1Affine::random(&mut thread_rng());
        let a_bytes = points_to_bytes(&vec![a]);
        let shader_code = load_bn254_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_identity(&shader_code, &a_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result.to_affine(), a);
    }

    #[test]
    fn test_scalar_mul() {
        let sample_size = 100;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);
        let fast = cpu_msm(&points, &scalars);

        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_bn254_point_msm_shader_code();
        let result =
            pollster::block_on(gpu::test::ops::point_msm(&shader_code, &p_bytes, &s_bytes));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        assert_eq!(point_result, fast);
    }

    #[test]
    fn test_sum_of_sums_simple() {
        let sample_size = 32;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);

        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_bn254_sum_of_sums_shader_code();
        let result = pollster::block_on(gpu::test::ops::sum_of_sums_simple(
            &shader_code,
            &p_bytes,
            &s_bytes,
        ));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(
            gpu_result[0].clone(),
            gpu_result[1].clone(),
            gpu_result[2].clone(),
        )
        .unwrap();
        println!("GPU Result: {:?}", point_result);
    }

    #[test]
    fn test_sum_of_sums() {
        let sample_size = 256;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);

        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_bn254_sum_of_sums_shader_code();
        let result = pollster::block_on(gpu::test::ops::sum_of_sums(
            &shader_code,
            &p_bytes,
            &s_bytes,
        ));
        let gpu_result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);

        let _ = gpu_result
            .chunks_exact(3)
            .map(|x| {
                let p = G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap();
                println!("GPU Result: {:?}", p);
                p
            })
            .collect::<Vec<_>>();
    }

    #[test]
    fn test_montgomery_repr() {
        let a = Fq::random(&mut thread_rng());
        let a_bytes = a.to_montgomery_repr();
        let a_result = Fq::from_montgomery_repr(a_bytes).unwrap();
        assert_eq!(a, a_result);
    }
}

#[cfg(test)]
mod tests_wasm_pack {
    use wasm_bindgen_test::*;
    use wasm_bindgen::prelude::*;
    // #[cfg(target_arch = "wasm32")]
    // use web_sys::console;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    // #[wasm_bindgen_test]
    // async fn test_webgpu_msm_bn254() {
    //     let sample_size = 80000;
    //     let points = sample_points::<G1Affine>(sample_size);
    //     let scalars = sample_scalars::<Fr>(sample_size);
        
    //     let cpu_start = now();
    //     let fast = cpu_msm(&points, &scalars);
    //     console::log_1(&format!("CPU Elapsed: {} ms", now() - cpu_start).into());

    //     let gpu_start = now();
    //     // let result = run_webgpu_msm_async_browser(&points, &scalars).await;
    //     let points_slice = points_to_bytes(&points);
    //     console::log_1(&format!("points_to_bytes: {} ms", now() - gpu_start).into());

    //     let vector_start = now();
    //     let v_slice = scalars_to_bytes(&scalars);
    //     console::log_1(&format!("scalars_to_bytes: {} ms", now() - vector_start).into());

    //     let shader_start = now();
    //     let shader_code = load_shader_code(Fq::MODULUS);
    //     console::log_1(&format!("load_shader_code: {} ms", now() - shader_start).into());

    //     let result_start = now();
    //     let result = gpu::msm::run_msm_browser(&shader_code, &points_slice, &v_slice).await;
    //     console::log_1(&format!("run_msm_browser: {} ms", now() - result_start).into());

    //     let result_start = now();
    //     let result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
    //     console::log_1(&format!("u16_vec_to_fields_montgomery: {} ms", now() - result_start).into());
    //     let result = G1::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap();
    //     console::log_1(&format!("GPU Elapsed: {} ms", now() - gpu_start).into());
        


    //     console::log_1(&format!("Result: {:?}", result).into());
    //     assert_eq!(fast, result);
    // }
}
