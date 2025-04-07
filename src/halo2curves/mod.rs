pub mod utils;
pub mod test;

use crate::{
    gpu, halo2curves::utils::fields_to_u16_vec, utils::{files::{load_shader_code, load_shader_code_bn254}, montgomery::{fields_to_u16_vec_montgomery, u16_vec_to_fields_montgomery}}
};
use ff::{Field, PrimeField};
use group::{Group, Curve};
use halo2curves::CurveExt;
use rand::thread_rng;

use halo2curves::{msm::best_multiexp, CurveAffine};



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
    fields_to_u16_vec(&v)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn fields_to_bytes_montgomery<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    fields_to_u16_vec_montgomery(&v)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn points_to_bytes<C: CurveAffine>(g: &Vec<C>) -> Vec<u8> {
    let packed_points: Vec<C::Base> = g
        .into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = coords.x();
            let y = coords.y();
            let z = C::Base::ONE;
            [*x, *y, z]
        })
        .collect::<Vec<_>>();
    fields_to_u16_vec_montgomery(&packed_points)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    pollster::block_on(run_webgpu_msm_async(g, v))
}

pub async fn run_webgpu_msm_async<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    let points_slice = points_to_bytes(g);
    let v_slice = scalars_to_bytes(v);
    let shader_code = load_shader_code(C::Base::MODULUS);
    let result = gpu::msm::run_msm(&shader_code, &points_slice, &v_slice).await;
    let result: Vec<<<C as CurveAffine>::CurveExt as CurveExt>::Base> = u16_vec_to_fields_montgomery(&result);
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
