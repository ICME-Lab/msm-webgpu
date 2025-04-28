use std::time::Instant;

use ff::PrimeField;
use wgpu::CommandEncoderDescriptor;

use crate::cuzk::{
    gpu::{
        create_and_write_storage_buffer, create_bind_group, create_bind_group_layout,
        create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter, get_device,
        read_from_gpu,
    },
    msm::{PARAMS, WORD_SIZE},
    shader_manager::ShaderManager,
    utils::{field_to_u8_vec_montgomery_for_gpu, u8s_to_field_without_assertion},
};

pub async fn field_op<F: PrimeField>(op: &str, a: F, b: F) -> F {
    let a_bytes = field_to_u8_vec_montgomery_for_gpu(&a, PARAMS.num_words, WORD_SIZE);
    let b_bytes = field_to_u8_vec_montgomery_for_gpu(&b, PARAMS.num_words, WORD_SIZE);

    let input_size = 1;
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_words = PARAMS.num_words;
    println!("Input size: {}", input_size);
    println!("Chunk size: {}", chunk_size);
    println!("Num words: {}", num_words);
    println!("Word size: {}", WORD_SIZE);
    println!("Params: {:?}", PARAMS);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size);

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
    let data = read_from_gpu(&device, &queue, encoder, vec![result_sb]).await;

    // Destroy the GPU device object.
    device.destroy();

    println!("Data: {:?}", data[0]);

    let result = u8s_to_field_without_assertion::<F>(&data[0], num_words, WORD_SIZE);

    result
}

pub fn run_webgpu_field_op<F: PrimeField>(op: &str, a: F, b: F) -> F {
    pollster::block_on(run_webgpu_field_op_async(op, a, b))
}

pub async fn run_webgpu_field_op_async<F: PrimeField>(op: &str, a: F, b: F) -> F {
    let now = Instant::now();
    let result = field_op::<F>(op, a, b).await;
    println!("Field add time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use halo2curves::bn256::Fq;
    use rand::thread_rng;

    #[test]
    fn test_webgpu_field_add() {
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let fast = a + b;

        let result = run_webgpu_field_op::<Fq>("test_field_add", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_field_sub() {
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let fast = a - b;

        let result = run_webgpu_field_op::<Fq>("test_field_sub", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_field_mul() {
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let fast = a * b;

        let result = run_webgpu_field_op::<Fq>("test_montgomery_product", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }
}
