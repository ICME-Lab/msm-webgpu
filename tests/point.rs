use std::time::Instant;

use halo2curves::{CurveAffine, CurveExt};
use wgpu::CommandEncoderDescriptor;

use msm_webgpu::cuzk::{
    gpu::{
        create_and_write_storage_buffer, create_and_write_uniform_buffer, create_bind_group,
        create_bind_group_layout, create_compute_pipeline, create_storage_buffer, execute_pipeline,
        get_adapter, get_device, read_from_gpu_test,
    },
    msm::WORD_SIZE,
    shader_manager::ShaderManager,
    utils::{bytes_to_field, compute_misc_params, compute_p, points_to_bytes_for_gpu, to_biguint_le},
};

async fn point_op<C: CurveAffine>(op: &str, a: C, b: C, scalar: u32) -> C::Curve {
    let p = compute_p::<C>();
    let params = compute_misc_params(&p, WORD_SIZE);
    let a_bytes = points_to_bytes_for_gpu(&[a], params.num_words, WORD_SIZE);
    let b_bytes = points_to_bytes_for_gpu(&[b], params.num_words, WORD_SIZE);
    let scalar_bytes = scalar.to_le_bytes();
    let input_size = 1;
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_words = params.num_words;
    println!("Input size: {input_size}");
    println!("Chunk size: {chunk_size}");
    println!("Num words: {num_words}");
    println!("Word size: {WORD_SIZE}");
    println!("Params: {params:?}");

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size, &params);

    let adapter = get_adapter().await;
    let (device, queue) = get_device(&adapter).await;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Point Encoder"),
    });

    let shader_code = shader_manager.gen_test_point_shader();

    let a_sb = create_and_write_storage_buffer(Some("A buffer"), &device, &a_bytes);
    let b_sb = create_and_write_storage_buffer(Some("B buffer"), &device, &b_bytes);

    let result_sb = create_storage_buffer(Some("Result buffer"), &device, 240);

    let scalar_sb =
        create_and_write_uniform_buffer(Some("Scalar buffer"), &device, &queue, &scalar_bytes);
    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        &device,
        vec![],
        vec![&a_sb, &b_sb, &result_sb],
        vec![&scalar_sb],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        &device,
        &bind_group_layout,
        vec![&a_sb, &b_sb, &result_sb, &scalar_sb],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Point shader"),
        &device,
        &bind_group_layout,
        &shader_code,
        op,
    )
    .await;

    execute_pipeline(&mut encoder, compute_pipeline, bind_group, 1, 1, 1).await;

    // Map results back from GPU to CPU.
    let data = read_from_gpu_test(&device, &queue, encoder, vec![result_sb]).await;

    // Destroy the GPU device object.
    device.destroy();

    let data_u32 = bytemuck::cast_slice::<u8, u32>(&data[0]);
    println!("Data u32: {data_u32:?}");
    println!("Data length: {:?}", data_u32.len());

    let results = data_u32
        .chunks(num_words)
        .map(|chunk| {
            let biguint_montgomery = to_biguint_le(chunk, num_words, WORD_SIZE as u32);
            let biguint = biguint_montgomery * &params.rinv % p.clone();
            let field: <<C as CurveAffine>::CurveExt as CurveExt>::Base =
                bytes_to_field(&biguint.to_bytes_le());
            field
        })
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    C::Curve::new_jacobian(results[0], results[1], results[2]).unwrap()
}

/// Run WebGPU point op sync
pub fn run_webgpu_point_op<C: CurveAffine>(op: &str, a: C, b: C, scalar: u32) -> C::Curve {
    pollster::block_on(run_webgpu_point_op_async(op, a, b, scalar))
}

/// Run WebGPU point op async
pub async fn run_webgpu_point_op_async<C: CurveAffine>(
    op: &str,
    a: C,
    b: C,
    scalar: u32,
) -> C::Curve {
    let now = Instant::now();
    let result = point_op::<C>(op, a, b, scalar).await;
    println!("Point op time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::{Field, PrimeField};
    use group::{Curve, Group};
    use group::cofactor::CofactorCurveAffine;
    use halo2curves::bn256::{Fr, G1Affine};
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Point as PallasPoint, Scalar as PallasScalar, Base as PallasBase};
    use halo2curves::secp256k1::{Fq as Secp256k1Fq, Secp256k1, Secp256k1Affine};
    use msm_webgpu::cuzk::utils::gen_p_limbs;
    use num_bigint::BigUint;
    use num_traits::Num;
    use rand::{Rng, thread_rng};

    fn test_webgpu_point_add<C: CurveAffine>() {
        let mut rng = thread_rng();
        let a = C::Curve::random(&mut rng).to_affine();
        println!("a: {:?}", a);
        let b = C::Curve::random(&mut rng).to_affine();
        println!("b: {:?}", b);

        let fast = a + b;

        let result = run_webgpu_point_op::<C>("test_point_add", a, b, 0);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_point_add_bn256() {
        test_webgpu_point_add::<G1Affine>();
    }

    #[test]
    fn test_webgpu_point_add_pallas() {
        test_webgpu_point_add::<PallasAffine>();
    }

    #[test]
    fn test_webgpu_point_add_secp256k1() {
        test_webgpu_point_add::<Secp256k1Affine>();
    }

    fn test_webgpu_point_add_identity<C: CurveAffine>() {
        let mut rng = thread_rng();
        let a = C::Curve::random(&mut rng).to_affine();
        println!("a: {:?}", a);
        let b = C::identity();
        println!("b: {:?}", b);

        let fast = a + b;

        let result = run_webgpu_point_op::<C>("test_point_add_identity", a, b, 0);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_point_add_identity_bn256() {
        test_webgpu_point_add_identity::<G1Affine>();
    }

    #[test]
    fn test_webgpu_point_add_identity_pallas() {
        test_webgpu_point_add_identity::<PallasAffine>();
    }

    #[test]
    fn test_webgpu_point_add_identity_secp256k1() {
        test_webgpu_point_add_identity::<Secp256k1Affine>();
    }


    fn test_webgpu_point_negate<C: CurveAffine>() {

        for _ in 0..1000 {
        let mut rng = thread_rng();
        let a = C::Curve::random(&mut rng).to_affine();
        println!("a: {:?}", a);

        let fast = -a;

        let result = run_webgpu_point_op::<C>("test_negate_point", a, a, 0);

        println!("Result: {:?}", result);
        assert_eq!(fast, result.to_affine());
        }
    }

    #[test]
    fn test_webgpu_point_negate_bn256() {
        test_webgpu_point_negate::<G1Affine>();
    }

    #[test]
    fn test_webgpu_point_negate_pallas() {
        test_webgpu_point_negate::<PallasAffine>();
    }

    #[test]
    fn test_webgpu_point_negate_secp256k1() {
        test_webgpu_point_negate::<Secp256k1Affine>();
    }

    fn test_webgpu_point_double_and_add<C: CurveAffine>() {
        let mut rng = thread_rng();
        let a = C::Curve::random(&mut rng).to_affine();
        println!("a: {:?}", a);
        // random u32
        let scalar = rng.gen_range(0..u32::MAX);
        println!("scalar: {:?}", scalar);

        let fast = a * C::Scalar::from(scalar as u64);

        let result = run_webgpu_point_op::<C>("test_double_and_add", a, a, scalar);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_point_double_and_add_bn256() {
        test_webgpu_point_double_and_add::<G1Affine>();
    }

    #[test]
    fn test_webgpu_point_double_and_add_pallas() {
        test_webgpu_point_double_and_add::<PallasAffine>();
    }

    #[test]
    fn test_webgpu_point_double_and_add_secp256k1() {
        test_webgpu_point_double_and_add::<Secp256k1Affine>();
    }
}
