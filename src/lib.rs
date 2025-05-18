pub mod cuzk;

use crate::cuzk::msm::compute_msm;
use ff::PrimeField;
use group::{Curve, Group};
use rand::thread_rng;

use halo2curves::{msm::best_multiexp, CurveAffine};

use crate::cuzk::utils::field_to_bytes;

/// Sample random scalars
pub fn sample_scalars<F: PrimeField>(n: usize) -> Vec<F> {
    let mut rng = thread_rng();
    (0..n).map(|_| F::random(&mut rng)).collect::<Vec<_>>()
}

/// Sample random affine points
pub fn sample_points<C: CurveAffine>(n: usize) -> Vec<C> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| C::Curve::random(&mut rng).to_affine())
        .collect::<Vec<_>>()
}

/// Run CPU MSM computation
pub fn cpu_msm<C: CurveAffine>(g: &[C], v: &[C::Scalar]) -> C::Curve {
    best_multiexp(v, g)
}

/// Convert scalars to bytes 
pub fn scalars_to_bytes<F: PrimeField>(v: &[F]) -> Vec<u8> {
    v.iter().flat_map(|x| field_to_bytes(x)).collect::<Vec<_>>()
}

/// Convert points to bytes as [x0, y0, x1, y1, ...]
pub fn points_to_bytes<C: CurveAffine>(g: &[C]) -> Vec<u8> {
    g.into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_bytes(coords.x());
            let y = field_to_bytes(coords.y());
            [x, y].concat()
        })
        .collect::<Vec<_>>()
}

#[cfg(not(target_arch = "wasm32"))]
/// Run WebGPU MSM computation synchronously
pub fn run_webgpu_msm<C: CurveAffine>(g: &[C], v: &[C::Scalar]) -> C::Curve {
    pollster::block_on(compute_msm(g, v))
}

#[cfg(target_arch = "wasm32")]
/// Run WebGPU MSM computation asynchronously
pub async fn run_webgpu_msm<C: CurveAffine>(g: &[C], v: &[C::Scalar]) -> C::Curve {
    compute_msm(g, v).await
}


pub mod tests_wasm_pack {
    use crate::cuzk::msm::compute_msm;

    use super::*;

    use halo2curves::bn256::{Fr, G1Affine};
    use wasm_bindgen::prelude::*;
    use web_sys::console;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    pub async fn test_webgpu_msm_cuzk(sample_size: usize) {
        console::log_1(&format!("Testing with sample size: {}", sample_size).into());
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);

        let cpu_start = now();
        let fast = cpu_msm(&points, &scalars);
        console::log_1(&format!("CPU Elapsed: {} ms", now() - cpu_start).into());

        let result_start = now();
        let result = compute_msm::<G1Affine>(&points, &scalars).await;
        console::log_1(&format!("GPU Elapsed: {} ms", now() - result_start).into());

        console::log_1(&format!("Result: {:?}", result).into());
        assert_eq!(fast, result);
    }


}
