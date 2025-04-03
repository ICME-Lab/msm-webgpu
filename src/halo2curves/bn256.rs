use crate::{
    gpu, halo2curves::utils::{fields_to_u16_vec, u16_vec_to_fields}, montgomery::{utils::{fields_to_u32_vec_montgomery, u32_vec_to_fields_montgomery}, MontgomeryRepr}, utils::load_shader_code_bn254
};
use ff::{Field, PrimeField};
use group::{Group, Curve};
use halo2curves::{
    bn256::{Fq, Fr, G1Affine, G1},
    CurveExt,
};
use rand::thread_rng;

use halo2curves::{msm::best_multiexp, CurveAffine};

use crate::montgomery::utils::{fields_to_u16_vec_montgomery, u16_vec_to_fields_montgomery};

use super::utils::fields_to_u32_vec;


pub fn sample_scalars(n: usize) -> Vec<Fr> {
    let mut rng = thread_rng();
    (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>()
}

pub fn sample_points(n: usize) -> Vec<G1Affine> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| G1Affine::random(&mut rng))
        .collect::<Vec<_>>()
}

pub fn slow_msm(g: &Vec<G1Affine>, v: &Vec<Fr>) -> G1 {
    let mut acc = G1::identity();
    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    acc
}

pub fn fast_msm(g: &Vec<G1Affine>, v: &Vec<Fr>) -> G1 {
    best_multiexp(v, g)
}

pub fn scalars_to_bytes<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    fields_to_u16_vec(&v)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn fields_to_bytes_montgomery<F: MontgomeryRepr>(v: &Vec<F>) -> Vec<u8> {
    fields_to_u16_vec_montgomery(&v)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn points_to_bytes(g: &Vec<G1Affine>) -> Vec<u8> {
    let packed_points: Vec<Fq> = g
        .into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = coords.x();
            let y = coords.y();
            let z = Fq::one();
            [*x, *y, z]
        })
        .collect::<Vec<_>>();
    fields_to_u16_vec_montgomery(&packed_points)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm(g: &Vec<G1Affine>, v: &Vec<Fr>) -> G1 {
    pollster::block_on(run_webgpu_msm_async(g, v))
}

pub async fn run_webgpu_msm_async(g: &Vec<G1Affine>, v: &Vec<Fr>) -> G1 {
    let points_slice = points_to_bytes(g);
    let v_slice = scalars_to_bytes(v);
    let shader_code = load_shader_code_bn254();
    let result = gpu::msm::run_msm(&shader_code, &points_slice, &v_slice).await;
    let result: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
    println!("Result: {:?}", (result[0], result[1], result[2]));
    G1::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use group::cofactor::CofactorCurveAffine;
    use rand::Rng;

    use crate::gpu::test::pippenger::{emulate_bucket_accumulation, emulate_bucket_reduction, emulate_pippenger, emulate_pippenger_gpu};
    use crate::montgomery::utils::field_to_bytes_montgomery;
    use crate::montgomery::MontgomeryRepr;

    use crate::halo2curves::utils::{cast_u8_to_u16, cast_u8_to_u32, u32_vec_to_fields};

    use super::*;

    #[test]
    fn test_bn256() {
        let sample_size = 1000;
        let scalars = sample_scalars(sample_size);
        let points = sample_points(sample_size);

        let now = Instant::now();
        let slow = slow_msm(&points, &scalars);
        println!("Slow Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now();
        let fast = fast_msm(&points, &scalars);
        println!("Fast Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now();
        let webgpu = run_webgpu_msm(&points, &scalars);
        println!("WebGPU Elapsed: {:.2?}", now.elapsed());
        assert_eq!(slow, fast);
        assert_eq!(fast, webgpu);
    }

    #[test]
    fn test_pippenger_emul() {
        let sample_size = 1000;
        let scalars = sample_scalars(sample_size);
        let points = sample_points(sample_size);
        let now = Instant::now();
        let fast = fast_msm(&points, &scalars);
        println!("Fast Elapsed: {:.2?}", now.elapsed());
        let now = Instant::now()    ;
        let result = emulate_pippenger_gpu(&points, &scalars);
        println!("WebGPU Elapsed: {:.2?}", now.elapsed());

        assert_eq!(result, fast);
    }



    #[test]
    fn test_fields_to_u16_vec() {
        let fields = sample_scalars(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_fields_to_u32_vec() {
        let fields = sample_scalars(10);
        let u32_vec = fields_to_u32_vec(&fields);
        let new_fields = u32_vec_to_fields(&u32_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_u16_field_casting() {
        let fields = sample_scalars(10);
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
        let fields = sample_scalars(10);
        let u32_vec = fields_to_u32_vec(&fields);
        let u32_vec_as_u8 = u32_vec
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        let casted_32 = cast_u8_to_u32(&u32_vec_as_u8);
        let new_fields = u32_vec_to_fields(&casted_32);
        assert_eq!(fields, new_fields);
    }


    fn load_field_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/field.wgsl"));
        shader_code
    }

    fn load_point_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/point.wgsl"));
        shader_code
    }

    fn load_point_msm_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/msm.wgsl"));
        shader_code
    }

    fn load_sum_of_sums_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/sum_of_sums.wgsl"));
        shader_code
    }

    fn load_field_to_bytes_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/field_to_bytes.wgsl"));
        shader_code
    }

    fn load_constants_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/constants.wgsl"));
        shader_code
    }

    fn load_pippenger_phases_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pippenger.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/pippenger_phases.wgsl"));
        shader_code
    }

    #[test]
    fn test_field_to_bytes() {
        let field = Fq::random(&mut thread_rng());
        let bytes = fields_to_bytes_montgomery(&vec![field]);
        let shader_code = load_field_to_bytes_shader_code();
        let gpu_bytes = pollster::block_on(gpu::test::ops::field_to_bytes(&shader_code, &bytes));
        let bytes = field_to_bytes_montgomery(field);
        let gpu_bytes = cast_u8_to_u32(&gpu_bytes).iter().map(|x| *x as u8).collect::<Vec<_>>();
        assert_eq!(bytes, gpu_bytes);
    }

    #[test]
    fn test_constants() {
        let random_size = thread_rng().gen_range(1..1000);
        let fields = sample_scalars(random_size);
        let bytes = scalars_to_bytes(&fields);
        let shader_code = load_constants_shader_code();
        let result = pollster::block_on(gpu::test::constants::run_constants(&shader_code, &bytes));
        println!("Result: {:?}", result);
        assert_eq!(result[0], fields.len() as u32);
        let num_invocations = (fields.len() + 64 - 1) / 64;
        assert_eq!(result[1], num_invocations as u32);
    }

    #[test]
    fn test_bucket_accumulation() {
        let sample_size = 64;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);
        let points_bytes = points_to_bytes(&points);
        let scalars_bytes = scalars_to_bytes(&scalars);
        let shader_code = load_pippenger_phases_shader_code();
        let result = pollster::block_on(gpu::test::pippenger_phases::run_bucket_accumulation(&shader_code, &points_bytes, &scalars_bytes));
        let gpu_buckets_in_fields: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let gpu_buckets = gpu_buckets_in_fields.chunks_exact(3).map(|x| {
            G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap()
        }).collect::<Vec<_>>();
        let num_invocations = (points.len() + 64 - 1) / 64;
        let total_buckets = 8192;
        let mut emulated_buckets = vec![G1::identity(); total_buckets * num_invocations];
        for i in 0..num_invocations {
            emulate_bucket_accumulation(&points, &scalars, &mut emulated_buckets, i);
        }
        for (i, (gpu_bucket, emulated_bucket)) in gpu_buckets.iter().zip(emulated_buckets.iter()).enumerate() {
            if gpu_bucket.is_identity().unwrap_u8() == 0 || emulated_bucket.is_identity().unwrap_u8() == 0 {
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
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);
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
        let shader_code = load_pippenger_phases_shader_code();
        let result = pollster::block_on(gpu::test::pippenger_phases::run_bucket_reduction(&shader_code, &points_bytes, &scalars_bytes));
        let gpu_windows_in_fields: Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let gpu_windows = gpu_windows_in_fields.chunks_exact(3).enumerate().map(|(i, x)| {
            let p = G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap();
            if p.is_identity().unwrap_u8() == 0 {
                println!("Index: {:?}", i);
                println!("GPU Window: {:?}", p);
            }
            p
        }).collect::<Vec<_>>();
        for (i, (gpu_window, emulated_window)) in gpu_windows.iter().zip(emulated_windows.iter()).enumerate() {
            if gpu_window.is_identity().unwrap_u8() == 0 || emulated_window.is_identity().unwrap_u8() == 0 {
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

        let shader_code = load_field_shader_code();

        let result = pollster::block_on(gpu::test::ops::field_mul(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    } 

    #[test]
    fn test_field_add() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a + b;

        let a_bytes = fields_to_bytes_montgomery(&vec![a]);
        let b_bytes = fields_to_bytes_montgomery(&vec![b]);

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_sub() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());  
        let c = a - b;

        let a_bytes = fields_to_bytes_montgomery(&vec![a]);
        let b_bytes = fields_to_bytes_montgomery(&vec![b]);

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_sub(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_point_add() {
        let a = G1Affine::random(&mut thread_rng());
        let b = G1Affine::random(&mut thread_rng());
        let c = a + b;
        
        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_add_identity_right() {
        let a = G1Affine::random(&mut thread_rng());
        let b = G1Affine::identity();
        
        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        println!("GPU Result: {:?}", gpu_result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, G1::from(a));
    }

    #[test]
    fn test_point_add_identity_left() {
        let a = G1Affine::identity();
        let b = G1Affine::random(&mut thread_rng());
        
        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, G1::from(b));
    }


    #[test]
    fn test_point_double() {
        let a = G1Affine::random(&mut thread_rng());
        let c = a + a;
        
        let a_bytes = points_to_bytes(&vec![a]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_double(&shader_code, &a_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_identity() {
        let a = G1Affine::random(&mut thread_rng());
        let a_bytes = points_to_bytes(&vec![a]);
        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_identity(&shader_code, &a_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result.to_affine(), a);
    }

    #[test]
    fn test_scalar_mul() {
        let sample_size = 100;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);
        let fast = fast_msm(&points, &scalars);
        
        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_point_msm_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_msm(&shader_code, &p_bytes, &s_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, fast);
    }

    #[test]
    fn test_sum_of_sums_simple() {
        let sample_size = 32;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);
        
        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_sum_of_sums_shader_code();
        let result = pollster::block_on(gpu::test::ops::sum_of_sums_simple(&shader_code, &p_bytes, &s_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);
        let point_result = G1::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        println!("GPU Result: {:?}", point_result);
    }

    #[test]
    fn test_sum_of_sums() {
        let sample_size = 256;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);
        
        let p_bytes = points_to_bytes(&points);
        let s_bytes = scalars_to_bytes(&scalars);

        let shader_code = load_sum_of_sums_shader_code();
        let result = pollster::block_on(gpu::test::ops::sum_of_sums(&shader_code, &p_bytes, &s_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields_montgomery(&result);

        let _ = gpu_result.chunks_exact(3).map(|x| {
            let p = G1::new_jacobian(x[0].clone(), x[1].clone(), x[2].clone()).unwrap();
            println!("GPU Result: {:?}", p);
            p
        }).collect::<Vec<_>>();
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
    use super::*;
    use wasm_bindgen_test::*;
    use web_sys::console;
  
    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_bn254() {
        let sample_size = 5;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);

        let fast = fast_msm(&points, &scalars);
        let result = run_webgpu_msm_async(&points, &scalars).await;
        console::log_1(&format!("Result: {:?}", result).into());
        assert_eq!(fast, result);
    }
}