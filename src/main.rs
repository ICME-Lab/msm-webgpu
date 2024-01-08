use ark_bn254::{G1Projective, Fr, g1::Config, Fq};
use ark_ec::{ScalarMul, VariableBaseMSM, CurveGroup, short_weierstrass::Projective};
use ark_ff::{UniformRand};
use msm_webgpu::{
    gpu,
    utils::{bigints_to_bytes, concat_files, point_to_bytes, u32s_to_bigints, fr_vec_to_biguint_vec},
};
use num_bigint::BigUint;
use std::{time::Instant};
use ark_std::{Zero, One};

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
        .map(|_| G1Projective::rand(&mut rng))
        .collect::<Vec<_>>();

    let g = G1Projective::batch_convert_to_mul_base(&g);
    let mut acc = G1Projective::zero();

    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    let fast = G1Projective::msm(g.as_slice(), v.as_slice()).unwrap();

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

    let ans = Projective::<Config>::new_unchecked(result[0].clone().try_into().expect("failed"),result[1].clone().try_into().expect("failed"),result[2].clone().try_into().expect("failed"));

    assert_eq!(fast.into_affine(), ans.into_affine());
}
