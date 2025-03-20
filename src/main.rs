use ark_pallas::{Fq, Fr, Projective as PallasProjective};
use ark_ec::{short_weierstrass::Projective, CurveGroup, PrimeGroup, ScalarMul, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::{One, Zero};
use msm_webgpu::{
    gpu,
    utils::{
        bigints_to_bytes, concat_files, fields_to_u16_vec, fr_vec_to_biguint_vec, point_to_bytes, u16_array_to_big_int, u16_array_to_fields, u32s_to_bigints
    },
};
use num_bigint::BigUint;
use std::{io::Read, time::Instant};

fn main() {
    /*
       SETUP
    */
    const SAMPLES: usize = 8;
    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    // let v = (0..SAMPLES).map(|_| Fr::from(2)).collect::<Vec<_>>();
    println!("v: {:?}", v);
    let bigint_v = fr_vec_to_biguint_vec(&v);
    let v_slice = bigints_to_bytes(bigint_v.clone());

    // let g = (0..SAMPLES)
    //     .map(|_| PallasProjective::rand(&mut rng))
    //     .collect::<Vec<_>>();

    let g = (0..SAMPLES)
        .map(|_| PallasProjective::generator())
        .collect::<Vec<_>>();

    let g = PallasProjective::batch_convert_to_mul_base(&g);
    println!("g: {:?}", g);
    let mut acc = PallasProjective::zero();

    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    let fast = PallasProjective::msm(g.as_slice(), v.as_slice()).unwrap();

    let packed_points: Vec<Fq> = g
        .into_iter()
        .flat_map(|affine| [affine.x, affine.y, Fq::one()])
        .collect::<Vec<_>>();
    let points_slice = fields_to_u16_vec(&packed_points)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>();

    let output_u32: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&points_slice).to_vec();
    let output_u16 = output_u32.iter().map(|&x| {
        if x > u16::MAX as u32 {
            panic!("Value {} is too large for u16", x);
        }
        x as u16
    }).collect::<Vec<_>>();
    let result: Vec<Fq> = u16_array_to_fields(&output_u16);
    println!("result: {:?}", result);

    assert_eq!(result, packed_points);
    /*
    RUN
    */
    let now = Instant::now();
    let shader_code = concat_files(vec!["src/wgsl/all.wgsl"]);
    let result = pollster::block_on(gpu::run_msm_compute(
        &shader_code,
        &points_slice,
        &v_slice,
        // 3 * 64,
    ));

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let result: Vec<Fq> = u16_array_to_fields(&result);

    println!("result: {:?}", result.len());

    let ans = PallasProjective::new_unchecked(
        result[0].clone(),
        result[1].clone(),
        result[2].clone(),
    );

    assert_eq!(fast.into_affine(), ans.into_affine());
}
