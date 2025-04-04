


#[cfg(test)]
mod tests {
    use std::time::Instant;
    use ff::{Field, PrimeField};
    use group::{Group, Curve};
    use halo2curves::{pasta::pallas::{Affine, Base as Fq, Point, Scalar as Fr}, CurveExt};
    use rand::thread_rng;
    use crate::{ gpu, halo2curves::utils::{fields_to_u16_vec, u16_vec_to_fields}, utils::files::{load_pallas_field_shader_code, load_pallas_point_msm_shader_code, load_pallas_point_shader_code, load_shader_code_pallas}
    };
    use crate::halo2curves::*;
    
    
    

  
    #[test]
    fn test_pallas() {
        let sample_size = 64;
        let scalars = sample_scalars::<Fr>(sample_size);
        let points = sample_points::<Affine>(sample_size);

        let now = Instant::now();
        let fast = fast_msm(&points, &scalars);
        println!("Fast Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now();
        let webgpu = run_webgpu_msm(&points, &scalars);
        println!("WebGPU Elapsed: {:.2?}", now.elapsed());
        assert_eq!(fast, webgpu);
    }

    #[test]
    fn test_fields_to_u16_vec() {
        let fields = sample_scalars::<Fq>(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_field_casting() {
        let fields = sample_scalars::<Fq>(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let u16_vec_as_u8 = u16_vec
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        let casted_32 = bytemuck::cast_slice::<u8, u32>(&u16_vec_as_u8).to_vec();
        let casted_16 = casted_32
            .iter()
            .map(|&x| {
                if x > u16::MAX as u32 {
                    panic!("Value {} is too large for u16", x);
                }
                x as u16
            })
            .collect::<Vec<_>>();
        let new_fields = u16_vec_to_fields(&casted_16);
        assert_eq!(fields, new_fields);
    }



    #[test]
    fn test_field_mul() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a * b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_pallas_field_shader_code();

        let result = pollster::block_on(gpu::test::ops::field_mul(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_add() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a + b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_pallas_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_sub() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());  
        let c = a - b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_pallas_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_sub(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }


    #[test]
    fn test_point_add() {
        let a = Point::random(&mut thread_rng()).to_affine();
        let b = Point::random(&mut thread_rng()).to_affine();
        let c = a + b;
        
        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_pallas_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_double() {
        let a = Point::random(&mut thread_rng()).to_affine();
        let c = a + a;
        
        let a_bytes = points_to_bytes(&vec![a]);

        let shader_code = load_pallas_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_double(&shader_code, &a_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_scalar_mul() {
        let p = Point::random(&mut thread_rng()).to_affine();
        let s = Fr::random(&mut thread_rng());
        let c = p * s;
        
        let p_bytes = points_to_bytes(&vec![p]);
        let s_bytes = scalars_to_bytes(&vec![s]);

        let shader_code = load_pallas_point_msm_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_msm(&shader_code, &p_bytes, &s_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }
}

// #[cfg(test)]
// mod tests_wasm_pack {
//     use halo2curves::pasta::pallas::{Affine, Scalar as Fr};
//     use wasm_bindgen_test::*;
//     use web_sys::console;
//     use crate::halo2curves::*;
//     wasm_bindgen_test_configure!(run_in_browser);

//     #[wasm_bindgen_test]
//     async fn test_webgpu_msm_pallas() {
//         let sample_size = 5;
//         let points = sample_points::<Affine>(sample_size);
//         let scalars = sample_scalars::<Fr>(sample_size);

//         let fast = fast_msm(&points, &scalars);
//         let result = run_webgpu_msm_async_browser(&points, &scalars).await;
//         console::log_1(&format!("Result: {:?}", result).into());
//         assert_eq!(fast, result);
//     }
// }
