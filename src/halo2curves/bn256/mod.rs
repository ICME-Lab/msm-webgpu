use crate::{
    gpu,
    halo2curves::utils::{fields_to_u16_vec, u16_vec_to_fields}, utils::load_shader_code_bn254,
};
use ff::{Field, PrimeField};
use group::Group;
use halo2curves::{
    bn256::{Fq, Fr, G1Affine, G1},
    CurveExt,
};
use rand::thread_rng;

use halo2curves::{msm::best_multiexp, CurveAffine};


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
    fields_to_u16_vec(&packed_points)
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
    let result: Vec<Fq> = u16_vec_to_fields(&result);
    G1::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::halo2curves::utils::cast_u8_to_u16;

    use super::*;

    #[test]
    fn test_bn254() {
        let sample_size = 5;
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
    fn test_fields_to_u16_vec() {
        let fields = sample_scalars(10);
        let u16_vec = fields_to_u16_vec(&fields);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_field_casting() {
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

    fn load_field_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../../wgsl/bn254/field.wgsl"));
        shader_code.push_str(include_str!("../../wgsl/test/ops.wgsl"));
        shader_code
    }

    #[test]
    fn test_field_mul() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a * b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_field_shader_code();

        let result = pollster::block_on(gpu::ops::field_mul(&shader_code, &a_bytes, &b_bytes));
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

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::ops::field_add(&shader_code, &a_bytes, &b_bytes));
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

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::ops::field_sub(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
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