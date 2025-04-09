pub mod utils;
pub mod test;

use std::time::Instant;

use crate::{
    gpu, halo2curves::utils::fields_to_u16_vec, utils::{files::{load_shader_code}, montgomery::{field_to_u16_as_u32_as_u8_vec_montgomery, fields_to_u16_vec_montgomery, u16_vec_to_fields_montgomery}}
};
use ff::{Field, PrimeField};
use group::{Group, Curve};
use halo2curves::CurveExt;
use rand::thread_rng;

use halo2curves::{msm::best_multiexp, CurveAffine};
use utils::field_to_u16_as_u32_as_u8_vec;



pub fn sample_scalars<F: PrimeField>(n: usize) -> Vec<F> {
    let mut rng = thread_rng();
    (0..n).map(|_| F::random(&mut rng)).collect::<Vec<_>>()
}

pub fn sample_points<C: CurveAffine>(n: usize) -> Vec<C> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| C::Curve::random(&mut rng).to_affine())
        .collect::<Vec<_>>()
}

pub fn fast_msm<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    best_multiexp(v, g)
}

pub fn scalars_to_bytes<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    v.iter().flat_map(|x| field_to_u16_as_u32_as_u8_vec(x)).collect::<Vec<_>>()
}

pub fn fields_to_bytes_montgomery<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    v.iter().flat_map(|x| field_to_u16_as_u32_as_u8_vec_montgomery(x)).collect::<Vec<_>>()
}

pub fn points_to_bytes<C: CurveAffine>(g: &Vec<C>) -> Vec<u8> {
    g
        .into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_u16_as_u32_as_u8_vec_montgomery(coords.x());
            let y = field_to_u16_as_u32_as_u8_vec_montgomery(coords.y());
            let z = field_to_u16_as_u32_as_u8_vec_montgomery(&C::Base::ONE);
            [x, y, z].concat()
        })
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    pollster::block_on(run_webgpu_msm_async(g, v))
}

pub async fn run_webgpu_msm_async<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    let now = Instant::now();
    let points_slice = points_to_bytes(g);
    println!("Points to bytes time: {:?}", now.elapsed());
    let now = Instant::now();
    let v_slice = scalars_to_bytes(v);
    println!("Scalars to bytes time: {:?}", now.elapsed());
    let now = Instant::now();
    let shader_code = load_shader_code(C::Base::MODULUS);
    println!("Shader code time: {:?}", now.elapsed());
    let result = gpu::msm::run_msm(&shader_code, &points_slice, &v_slice).await;
    let now = Instant::now();
    let result: Vec<<<C as CurveAffine>::CurveExt as CurveExt>::Base> = u16_vec_to_fields_montgomery(&result);
    println!("U16 to fields time: {:?}", now.elapsed());
    println!("Result: {:?}", (result[0], result[1], result[2]));
    C::Curve::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}

pub fn run_webgpu_msm_browser<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    pollster::block_on(run_webgpu_msm_async_browser(g, v))
}

pub async fn run_webgpu_msm_async_browser<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    let points_slice = points_to_bytes(g);
    let v_slice = scalars_to_bytes(v);
    let shader_code = load_shader_code(C::Base::MODULUS);
    let result = gpu::msm::run_msm_browser(&shader_code, &points_slice, &v_slice).await;
    let result: Vec<<<C as CurveAffine>::CurveExt as CurveExt>::Base> = u16_vec_to_fields_montgomery(&result);
    println!("Result: {:?}", (result[0], result[1], result[2]));
    C::Curve::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::{bn256::{Fq, G1Affine}, CurveAffine};

    #[test]
    fn test_points_to_bytes() {
        let g = sample_points::<G1Affine>(10);
        let bytes = points_to_bytes(&g);

        let packed_points: Vec<Fq> = g
        .into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = coords.x();
            let y = coords.y();
            let z = Fq::ONE;
            [*x, *y, z]
        })
        .collect::<Vec<_>>();
        let bytes_packed = fields_to_u16_vec_montgomery(&packed_points)
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(bytes, bytes_packed);
    }

    #[test]
    fn test_fields_to_bytes_montgomery() {
        let v = sample_scalars::<Fq>(10);
        let bytes = fields_to_bytes_montgomery(&v);
        let bytes_packed = fields_to_u16_vec_montgomery(&v)
            .into_iter()
            .flat_map(|x| (x as u32).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(bytes, bytes_packed);
    }
}
