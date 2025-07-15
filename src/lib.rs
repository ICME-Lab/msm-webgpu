#![allow(clippy::too_many_arguments)]

pub mod cuzk;

use crate::cuzk::msm::compute_msm;
use cuzk::utils::debug;
use ff::PrimeField;
use group::{Curve, Group};
use halo2curves::bn256::Fr;
use halo2curves::bn256::G1Affine;
use halo2curves::{msm::msm_best, CurveAffine};
use rand::thread_rng;
use js_sys::Array;
use rand::Rng;          

use crate::cuzk::utils::field_to_bytes;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures;

/// Sample random scalars
pub fn sample_scalars<F: PrimeField>(n: usize) -> Vec<F> {
    let mut rng = thread_rng();
    (0..n).map(|_| F::random(&mut rng)).collect::<Vec<_>>()
}

/// Sample random scalars
pub fn sample_32_bit_scalars<F: PrimeField>(n: usize) -> Vec<F> {
    let mut rng = thread_rng();

    (0..n).map(|_| {
        let random_u32: u32 = rng.gen_range(0..=u32::MAX);
        F::from(random_u32 as u64)
    }).collect::<Vec<_>>()
}

/// Sample random affine points
pub fn sample_points<C: CurveAffine>(n: usize) -> Vec<C> {
    let mut rng = thread_rng();
    (0..n)
        // .map(|_| C::identity())
        .map(|_| C::Curve::random(&mut rng).to_affine())
        .collect::<Vec<_>>()
}

/// Run CPU MSM computation
pub fn cpu_msm<C: CurveAffine>(g: &[C], v: &[C::Scalar]) -> C::Curve {
    msm_best(v, g)
}

/// Convert scalars to bytes
pub fn scalars_to_bytes<F: PrimeField>(v: &[F]) -> Vec<u8> {
    v.iter().flat_map(|x| field_to_bytes(x)).collect::<Vec<_>>()
}

/// Convert points to bytes as [x0, y0, x1, y1, ...]
pub fn points_to_bytes<C: CurveAffine>(g: &[C]) -> Vec<u8> {
    let ps = g.iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_bytes(coords.x());
            let y = field_to_bytes(coords.y());
            [x, y].concat()
        })
        .collect::<Vec<_>>();
    ps
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}



/// Run WebGPU MSM computation asynchronously
pub async fn run_webgpu_msm<C: CurveAffine>(
    g: &[C],
    v: &[C::Scalar],
) -> C::Curve {
        let result = compute_msm(g, v).await;
        result
}

#[wasm_bindgen]
pub async fn run_webgpu_msm_web_bn256(
    sample_size: usize,
    _callback: js_sys::Function,
) -> Array {
    let start = now();
    debug(&format!("Testing with sample size: {sample_size}"));
    let points = sample_points::<G1Affine>(sample_size);
    let scalars = sample_scalars::<Fr>(sample_size);
    debug(&format!("Sampling points and scalars took {} ms", now() - start));

    let start = now();
    let result = compute_msm(&points, &scalars).await;
    let msm_elapsed = now() - start;
    debug(&format!("GPU MSM Elapsed: {} ms", msm_elapsed));
    let coords = result.to_affine().coordinates().unwrap();

    let x_str = format!("{:?}", coords.x());
    let y_str = format!("{:?}", coords.y());

    let arr = Array::new();
    arr.push(&JsValue::from(x_str));
    arr.push(&JsValue::from(y_str));
    arr.push(&JsValue::from(msm_elapsed));
    arr
}

#[wasm_bindgen]
pub async fn run_webgpu_msm_web_pallas(
    sample_size: usize,
    _callback: js_sys::Function,
) -> Array {
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Scalar as PallasScalar};
    let start = now();
    debug(&format!("Testing with sample size: {sample_size}"));
    let points = sample_points::<PallasAffine>(sample_size);
    let scalars = sample_scalars::<PallasScalar>(sample_size);
    debug(&format!("Sampling points and scalars took {} ms", now() - start));

    let start = now();
    let result = compute_msm(&points, &scalars).await;
    let msm_elapsed = now() - start;
    debug(&format!("GPU MSM Elapsed: {} ms", msm_elapsed));
    let coords = result.to_affine().coordinates().unwrap();

    let x_str = format!("{:?}", coords.x());
    let y_str = format!("{:?}", coords.y());

    let arr = Array::new();
    arr.push(&JsValue::from(x_str));
    arr.push(&JsValue::from(y_str));
    arr.push(&JsValue::from(msm_elapsed));
    arr
}

#[wasm_bindgen]
pub async fn run_cpu_msm_web_bn256(
    sample_size: usize,
    _callback: js_sys::Function,
) -> Array {
    let start = now();
    debug(&format!("Testing with sample size: {sample_size}"));
    let points = sample_points::<G1Affine>(sample_size);
    let scalars = sample_scalars::<Fr>(sample_size);
    debug(&format!("Sampling points and scalars took {} ms", now() - start));


    let start = now();
    let result = cpu_msm(&points, &scalars);
    let cpu_elapsed = now() - start;
    debug(&format!("CPU MSM Elapsed: {} ms", cpu_elapsed));
    let coords = result.to_affine().coordinates().unwrap();

    let x_str = format!("{:?}", coords.x());
    let y_str = format!("{:?}", coords.y());

    let arr = Array::new();
    arr.push(&JsValue::from(x_str));
    arr.push(&JsValue::from(y_str));
    arr.push(&JsValue::from(cpu_elapsed));
    arr
}


pub mod tests_wasm_pack {
    use super::*;

    use halo2curves::bn256::{Fr, G1Affine};
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Scalar as PallasScalar};
    use halo2curves::secp256k1::{Secp256k1Affine, Fq as Secp256k1Fq};
    use halo2curves::secq256k1::{Secq256k1Affine, Fq as Secq256k1Fq};
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    pub async fn test_webgpu_msm_cuzk<C: CurveAffine>(sample_size: usize) {
        debug(&format!("Testing with sample size: {sample_size}"));
        let points = sample_points::<C>(sample_size);
        let scalars = sample_scalars::<C::Scalar>(sample_size);

        let cpu_start = now();
        let fast = cpu_msm(&points, &scalars);
        debug(&format!("CPU Elapsed: {} ms", now() - cpu_start));

        let result_start = now();
        let result = run_webgpu_msm::<C>(&points, &scalars).await;
        debug(&format!("GPU Elapsed: {} ms", now() - result_start));

        debug(&format!("Result: {result:?}"));
        assert_eq!(fast, result);
    }

    pub async fn test_webgpu_msm_cuzk_bn256(sample_size: usize) {
        test_webgpu_msm_cuzk::<G1Affine>(sample_size).await;
    }

    pub async fn test_webgpu_msm_cuzk_pallas(sample_size: usize) {
        test_webgpu_msm_cuzk::<PallasAffine>(sample_size).await;
    }

    pub async fn test_webgpu_msm_cuzk_secp256k1(sample_size: usize) {
        test_webgpu_msm_cuzk::<Secp256k1Affine>(sample_size).await;
    }

    pub async fn test_webgpu_msm_cuzk_secq256k1(sample_size: usize) {
        test_webgpu_msm_cuzk::<Secq256k1Affine>(sample_size).await;
    }

}
