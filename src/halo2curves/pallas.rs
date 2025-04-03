use ff::{Field, PrimeField};
use group::{Group, Curve};
use halo2curves::{pasta::pallas::{Affine, Base as Fq, Point, Scalar as Fr}, CurveExt};
use num_traits::One;
use rand::thread_rng;
use crate::{ gpu, halo2curves::utils::{fields_to_u16_vec, u16_vec_to_fields}, utils::load_shader_code_pallas
};

use halo2curves::{msm::best_multiexp, CurveAffine};



pub fn sample_scalars(n: usize) -> Vec<Fr> {
    let mut rng = thread_rng();
    (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>()
}

pub fn sample_points(n: usize) -> Vec<Affine> {
    let mut rng = thread_rng();
    (0..n)
        .map(|_| Point::random(&mut rng).to_affine())
        .collect::<Vec<_>>()
}

pub fn slow_msm(g: &Vec<Affine>, v: &Vec<Fr>) -> Point {
    let mut acc = Point::identity();
    for (base, scalar) in g.iter().zip(v.iter()) {
        acc += *base * scalar;
    }
    acc
}

pub fn fast_msm(g: &Vec<Affine>, v: &Vec<Fr>) -> Point {
    best_multiexp(v, g)
}

pub fn scalars_to_bytes<F: PrimeField>(v: &Vec<F>) -> Vec<u8> {
    fields_to_u16_vec(&v)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn points_to_bytes(g: &Vec<Affine>) -> Vec<u8> {
    let packed_points: Vec<Fq> = g
        .into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = coords.x();
            let y = coords.y();
            let z = Fq::one();
            [*x, *y, z]
        })
        .collect::<Vec<_>>();
    fields_to_u16_vec(&packed_points)
        .into_iter()
        .flat_map(|x| (x as u32).to_le_bytes())
        .collect::<Vec<_>>()
}

pub fn run_webgpu_msm(g: &Vec<Affine>, v: &Vec<Fr>) -> Point {
    pollster::block_on(run_webgpu_msm_async(g, v))
}

pub async fn run_webgpu_msm_async(g: &Vec<Affine>, v: &Vec<Fr>) -> Point {
    let points_slice = points_to_bytes(g);
    let v_slice = scalars_to_bytes(v);
    let shader_code = load_shader_code_pallas();
    let result = gpu::msm::run_msm(&shader_code, &points_slice, &v_slice).await;
    let result: Vec<Fq> = u16_vec_to_fields(&result);
    println!("Result: {:?}", result);
    Point::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}   

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_ec::CurveGroup;
    use ark_ff::PrimeField;

    use crate::halo2curves::utils::{field_to_bytes, u32_vec_to_fields};

    use super::*;
  
    #[test]
    fn test_pallas() {
        let sample_size = 1;
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

    #[test]
    fn test_ark_vs_halo2() {
        use ark_pallas::{Fr as PallasFr, Fq as PallasFq, Projective as PallasPoint};
        use crate::ark::utils::fields_to_u16_vec as ark_fields_to_u16_vec;
        use crate::ark::pallas::{scalars_to_bytes as ark_scalars_to_bytes, points_to_bytes as ark_points_to_bytes};

        let halo2_scalars = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let halo2_generator = Point::generator();
        let (x,y,z) = halo2_generator.jacobian_coordinates();
        let halo2_points = (0..3).map(|_| Point::generator().to_affine()).collect::<Vec<_>>();

        let x_bytes = field_to_bytes(x);
        let y_bytes = field_to_bytes(y);
        let z_bytes = field_to_bytes(z);

        let ark_x = PallasFq::from_le_bytes_mod_order(&x_bytes);
        let ark_y = PallasFq::from_le_bytes_mod_order(&y_bytes);
        let ark_z = PallasFq::from_le_bytes_mod_order(&z_bytes);

        let ark_scalars = vec![PallasFr::from(1), PallasFr::from(2), PallasFr::from(3)];
        let ark_points = (0..3).map(|_| PallasPoint::new_unchecked(ark_x, ark_y, ark_z).into_affine()).collect::<Vec<_>>();

        let halo2_u16_fields = fields_to_u16_vec(&halo2_scalars);
        let ark_u16_fields = ark_fields_to_u16_vec(&ark_scalars);

        assert_eq!(halo2_u16_fields, ark_u16_fields);

        let halo2_scalar_bytes = scalars_to_bytes(&halo2_scalars);
        let ark_scalar_bytes = ark_scalars_to_bytes(&ark_scalars);

        assert_eq!(halo2_scalar_bytes.len(), ark_scalar_bytes.len());
        assert_eq!(halo2_scalar_bytes, ark_scalar_bytes);

        let halo2_point_bytes = points_to_bytes(&halo2_points);
        let ark_point_bytes = ark_points_to_bytes(&ark_points);

        assert_eq!(halo2_point_bytes, ark_point_bytes);
    }

    fn load_field_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/field.wgsl"));
        shader_code
    }

    #[test]
    fn test_field_mul() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a * b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_field_shader_code();

        let result = pollster::block_on(gpu::test::ops::field_mul(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_add() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());
        let c = a + b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }

    #[test]
    fn test_field_sub() {
        let a = Fq::random(&mut thread_rng());
        let b = Fq::random(&mut thread_rng());  
        let c = a - b;

        let a_bytes = scalars_to_bytes(&vec![a]);
        let b_bytes = scalars_to_bytes(&vec![b]);

        let shader_code = load_field_shader_code();
        let result = pollster::block_on(gpu::test::ops::field_sub(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        assert_eq!(gpu_result[0], c);
    }

    fn load_point_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pallas/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/point.wgsl"));
        shader_code
    }

    fn load_point_msm_shader_code() -> String {
        let mut shader_code = String::new();
        shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
        shader_code.push_str(include_str!("../wgsl/pallas/curve.wgsl"));
        shader_code.push_str(include_str!("../wgsl/test/msm.wgsl"));
        shader_code
    }

    #[test]
    fn test_point_add() {
        let a = Point::random(&mut thread_rng()).to_affine();
        let b = Point::random(&mut thread_rng()).to_affine();
        let c = a + b;
        
        let a_bytes = points_to_bytes(&vec![a]);
        let b_bytes = points_to_bytes(&vec![b]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_add(&shader_code, &a_bytes, &b_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_point_double() {
        let a = Point::random(&mut thread_rng()).to_affine();
        let c = a + a;
        
        let a_bytes = points_to_bytes(&vec![a]);

        let shader_code = load_point_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_double(&shader_code, &a_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }

    #[test]
    fn test_scalar_mul() {
        let p = Point::random(&mut thread_rng()).to_affine();
        let s = Fr::random(&mut thread_rng());
        let c = p * s;
        
        let p_bytes = points_to_bytes(&vec![p]);
        let s_bytes = scalars_to_bytes(&vec![s]);

        let shader_code = load_point_msm_shader_code();
        let result = pollster::block_on(gpu::test::ops::point_msm(&shader_code, &p_bytes, &s_bytes));
        let gpu_result : Vec<Fq> = u16_vec_to_fields(&result);
        let point_result = Point::new_jacobian(gpu_result[0].clone(), gpu_result[1].clone(), gpu_result[2].clone()).unwrap();
        assert_eq!(point_result, c);
    }
}

#[cfg(test)]
mod tests_wasm_pack {
    use super::*;
    use wasm_bindgen_test::*;
    use web_sys::console;
  
    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_pallas() {
        let sample_size = 5;
        let points = sample_points(sample_size);
        let scalars = sample_scalars(sample_size);

        let fast = fast_msm(&points, &scalars);
        let result = run_webgpu_msm_async(&points, &scalars).await;
        console::log_1(&format!("Result: {:?}", result).into());
        assert_eq!(fast, result);
    }
}
