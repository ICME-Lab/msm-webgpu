use std::time::Instant;

use crate::{
    cuzk::msm::compute_msm,
    gpu,
    halo2curves::utils::{field_to_bytes, field_to_u16_as_u32_as_u8_vec, fields_to_u16_vec},
    utils::{
        files::load_shader_code,
        montgomery::{
            field_to_bytes_montgomery, field_to_u16_as_u32_as_u8_vec_montgomery,
            fields_to_u16_vec_montgomery, u16_vec_to_fields_montgomery,
        },
    },
};
use ff::{Field, PrimeField};
use group::{Curve, Group};
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
    v.iter().flat_map(|x| field_to_bytes(x)).collect::<Vec<_>>()
}

pub fn fields_to_bytes_montgomery<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    v.iter()
        .flat_map(|x| field_to_bytes_montgomery(x))
        .collect::<Vec<_>>()
}

pub fn points_to_bytes<C: CurveAffine>(g: &Vec<C>) -> Vec<u8> {
    g.into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_bytes_montgomery(coords.x());
            let y = field_to_bytes_montgomery(coords.y());
            let z = field_to_bytes_montgomery(&C::Base::ONE);
            [x, y, z].concat()
        })
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm<C: CurveAffine>(g: &Vec<C>, v: &Vec<C::Scalar>) -> C::Curve {
    pollster::block_on(compute_msm(g, v))
}



#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fr, G1Affine};
    use super::*;

    #[test]
    fn test_webgpu_msm_cuzk() {
        let sample_size = 1;
        let points = sample_points::<G1Affine>(sample_size);
        let scalars = sample_scalars::<Fr>(sample_size);

        let fast = fast_msm(&points, &scalars);

        let result = run_webgpu_msm::<G1Affine>(&points, &scalars);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }
}
#[cfg(test)]
mod tests_wasm_pack {
    use crate::cuzk::msm::compute_msm;

    use super::*;

    use halo2curves::bn256::{Fr, G1Affine};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::*;
    use web_sys::console;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    // #[wasm_bindgen_test]
    // async fn test_webgpu_msm_cuzk() {
    //     let sample_size = 1;
    //     let points = sample_points::<G1Affine>(sample_size);
    //     let scalars = sample_scalars::<Fr>(sample_size);

    //     let cpu_start = now();
    //     let fast = fast_msm(&points, &scalars);
    //     console::log_1(&format!("CPU Elapsed: {} ms", now() - cpu_start).into());

    //     let gpu_start = now();
    //     let points_slice = points_to_bytes(&points);
    //     console::log_1(&format!("points_to_bytes: {} ms", now() - gpu_start).into());

    //     let vector_start = now();
    //     let v_slice = scalars_to_bytes(&scalars);
    //     console::log_1(&format!("scalars_to_bytes: {} ms", now() - vector_start).into());

    //     let result_start = now();
    //     let result = compute_msm::<G1Affine>(&points_slice, &v_slice).await;
    //     console::log_1(&format!("run_msm_browser: {} ms", now() - result_start).into());

    //     console::log_1(&format!("Result: {:?}", result).into());
    //     assert_eq!(fast, result);
    // }
}
