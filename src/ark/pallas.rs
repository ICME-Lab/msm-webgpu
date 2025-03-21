use ark_ec::short_weierstrass::Affine;
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_pallas::PallasConfig;
use ark_pallas::{Fq, Fr, Projective as PallasProjective};
use ark_std::{One, Zero};
use num_bigint::BigUint;
use crate::utils::load_shader_code_pallas;
use crate::{
    gpu,
    utils::{
        bigints_to_bytes,
    },
    ark::utils::{
        fields_to_u16_vec,
        u16_vec_to_fields,
    },
};


pub fn sample_scalars(n: usize) -> Vec<Fr> {
    let mut rng = ark_std::test_rng();
    (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>()
}

pub fn sample_points(n: usize) -> Vec<Affine<PallasConfig>> {
    let mut rng = ark_std::test_rng();
    (0..n)
        .map(|_| PallasProjective::rand(&mut rng).into_affine())
        .collect::<Vec<_>>()
}

pub fn slow_msm(g: &Vec<Affine<PallasConfig>>, v: &Vec<Fr>) -> PallasProjective {
    let mut acc = PallasProjective::zero();
    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    acc
}

pub fn fast_msm(g: &Vec<Affine<PallasConfig>>, v: &Vec<Fr>) -> PallasProjective {
    PallasProjective::msm(g.as_slice(), v.as_slice()).unwrap()
}

pub fn scalars_to_bytes(v: &Vec<Fr>) -> Vec<u8> {
    let bigint_v: Vec<BigUint> = v.iter().map(|v| (*v).into()).collect();
    bigints_to_bytes(bigint_v.clone())
}

pub fn points_to_bytes(g: &Vec<Affine<PallasConfig>>) -> Vec<u8> {
    let packed_points: Vec<Fq> = g
        .into_iter()
        .flat_map(|affine| [affine.x, affine.y, Fq::one()])
        .collect::<Vec<_>>();
    fields_to_u16_vec(&packed_points)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm(g: &Vec<Affine<PallasConfig>>, v: &Vec<Fr>) -> PallasProjective {
    let points_slice = points_to_bytes(g);
    let v_slice = scalars_to_bytes(v);
    let shader_code = load_shader_code_pallas();
    let result =
        pollster::block_on(gpu::run_msm_compute(&shader_code, &points_slice, &v_slice));
    let result: Vec<Fq> = u16_vec_to_fields(&result);
    PallasProjective::new_unchecked(result[0].clone(), result[1].clone(), result[2].clone())
}
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    #[test]
    fn test_pallas() {
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
}
