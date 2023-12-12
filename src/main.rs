use ark_curve25519::{Curve25519Config, EdwardsProjective as G1Projective, Fr};
use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    ScalarMul,
};
use ark_ff::UniformRand;
use msm_webgpu::{
    gpu,
    utils::{bigints_to_bytes, concat_files, point_to_bytes, split_biguint, u32s_to_bigints},
};
use num_bigint::{BigUint, ToBigInt};

pub fn fr_vec_to_biguint_vec(vals: &Vec<Fr>) -> Vec<BigUint> {
    vals.iter().map(|v| (*v).into()).collect()
}

fn main() {
    /*
       SETUP
    */
    const SAMPLES: usize = 10;
    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    let bigint_v = fr_vec_to_biguint_vec(&v);

    let v_slice = bigints_to_bytes(bigint_v.clone());

    let g = (0..SAMPLES)
        .map(|_| G1Projective::rand(&mut rng))
        .collect::<Vec<_>>();

    let g = G1Projective::batch_convert_to_mul_base(&g);

    let packed_points: Vec<(BigUint, BigUint, BigUint)> = g
        .into_iter()
        .map(|affine| (affine.x.into(), affine.y.into(), (1 as u32).into()))
        .collect();

    let points_slice = point_to_bytes(packed_points);

    /*
       RUN
    */
    let shader_code = concat_files(vec!["src/wgsl/main.wgsl"]);
    let result = pollster::block_on(gpu::run_msm_compute(
        &shader_code,
        &points_slice,
        &v_slice,
        3 * 64,
    ));
    let result = u32s_to_bigints(result);

    println!("{:?}", result);
}
