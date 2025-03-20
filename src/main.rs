use ark_pallas::{Fq, Fr, Projective as PallasProjective};
use ark_ec::{short_weierstrass::Projective, CurveGroup, ScalarMul, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::{One, Zero};
use msm_webgpu::{
    gpu,
    utils::{
        bigints_to_bytes, concat_files, fr_vec_to_biguint_vec, point_to_bytes, u32s_to_bigints,
    },
};
use num_bigint::BigUint;
use std::time::Instant;

fn main() {
    /*
       SETUP
    */
    const SAMPLES: usize = 8;
    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    let bigint_v = fr_vec_to_biguint_vec(&v);
    let v_slice = bigints_to_bytes(bigint_v.clone());

    let g = (0..SAMPLES)
        .map(|_| PallasProjective::rand(&mut rng))
        .collect::<Vec<_>>();

    let g = PallasProjective::batch_convert_to_mul_base(&g);
    let mut acc = PallasProjective::zero();

    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    let fast = PallasProjective::msm(g.as_slice(), v.as_slice()).unwrap();

    let packed_points: Vec<(BigUint, BigUint, BigUint)> = g
        .into_iter()
        .map(|affine| (affine.x.into(), affine.y.into(), Fq::one().into()))
        .collect();
    let points_slice = point_to_bytes(&packed_points);
    /*
    RUN
    */
    let now = Instant::now();
    let shader_code = concat_files(vec!["src/wgsl/main.wgsl"]);
    let result = pollster::block_on(gpu::run_msm_compute(
        &shader_code,
        &points_slice,
        &v_slice,
        3 * 64,
    ));

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let result = u32s_to_bigints(result);

    let ans = PallasProjective::new_unchecked(
        result[0].clone().try_into().expect("failed"),
        result[1].clone().try_into().expect("failed"),
        result[2].clone().try_into().expect("failed"),
    );

    assert_eq!(fast.into_affine(), ans.into_affine());
}
