use std::time::Instant;

use ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;
use wgpu::CommandEncoderDescriptor;

use msm_webgpu::cuzk::{
    gpu::{
        create_and_write_storage_buffer, create_bind_group, create_bind_group_layout,
        create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter, get_device,
        read_from_gpu_test,
    },
    msm::WORD_SIZE,
    shader_manager::ShaderManager,
    utils::{bytes_to_field, compute_misc_params, field_to_u8_vec_for_gpu, to_biguint_le},
};

async fn field_op<F: PrimeField>(op: &str, a: F, b: F) -> F {
    let p = BigUint::from_str_radix(&F::MODULUS[2..], 16).unwrap();
    let params = compute_misc_params(&p, WORD_SIZE);
    let a_bytes = field_to_u8_vec_for_gpu(&a, params.num_words, WORD_SIZE);
    let b_bytes = field_to_u8_vec_for_gpu(&b, params.num_words, WORD_SIZE);
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
        label: Some("Field Encoder"),
    });

    let shader_code = shader_manager.gen_test_field_shader();

    let a_sb = create_and_write_storage_buffer(Some("A buffer"), &device, &a_bytes);
    let b_sb = create_and_write_storage_buffer(Some("B buffer"), &device, &b_bytes);

    let result_sb = create_storage_buffer(Some("Result buffer"), &device, (num_words * 4) as u64);

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        &device,
        vec![],
        vec![&a_sb, &b_sb, &result_sb],
        vec![],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        &device,
        &bind_group_layout,
        vec![&a_sb, &b_sb, &result_sb],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Field add shader"),
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

    let result_biguint = to_biguint_le(data_u32, num_words, WORD_SIZE as u32);

    

    bytes_to_field(&result_biguint.to_bytes_le())
}

/// Run WebGPU field op sync
pub fn run_webgpu_field_op<F: PrimeField>(op: &str, a: F, b: F) -> F {
    pollster::block_on(run_webgpu_field_op_async(op, a, b))
}

/// Run WebGPU field op async
pub async fn run_webgpu_field_op_async<F: PrimeField>(op: &str, a: F, b: F) -> F {
    let now = Instant::now();
    let result = field_op::<F>(op, a, b).await;
    println!("Field add time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use msm_webgpu::{
        cuzk::utils::{calc_num_words, compute_p, u8s_to_field_without_assertion},
        sample_scalars,
    };

    use super::*;
    use ff::Field;
    use halo2curves::bn256::{Fq, G1Affine};
    use rand::thread_rng;

    #[test]
    fn test_webgpu_field_add() {
        let scalars = sample_scalars::<Fq>(50);
        for scalar in scalars.chunks(2) {
            let a = scalar[0];
            let b = scalar[1];

            let fast = a + b;

            let result = run_webgpu_field_op::<Fq>("test_field_add", a, b);

            println!("Result: {:?}", result);
            assert_eq!(fast, result);
        }
    }

    #[test]
    fn test_webgpu_field_sub() {
        let scalars = sample_scalars::<Fq>(50);
        for scalar in scalars.chunks(2) {
            let a = scalar[0];
            let b = scalar[1];

            let fast = a - b;

            let result = run_webgpu_field_op::<Fq>("test_field_sub", a, b);

            println!("Result: {:?}", result);
            assert_eq!(fast, result);
        }
    }

    #[test]
    fn test_webgpu_field_mul() {
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let fast = a * b;
        let result = run_webgpu_field_op::<Fq>("test_field_mul", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_field_barret_mul() {
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let fast = a;
        let result = run_webgpu_field_op::<Fq>("test_barret_mul", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_field_to_u8_vec_for_gpu() {
        let p = compute_p::<G1Affine>();
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        for word_size in 13..17 {
            let num_words = calc_num_words(&p, word_size);
            let bytes = field_to_u8_vec_for_gpu(&a, num_words, word_size);
            let a_from_bytes = u8s_to_field_without_assertion(&p, &bytes, num_words, word_size);
            assert_eq!(a, a_from_bytes);
        }
    }
}
