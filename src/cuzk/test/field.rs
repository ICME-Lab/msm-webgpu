use std::time::Instant;

use ff::PrimeField;
use wgpu::CommandEncoderDescriptor;

use crate::{
    cuzk::{
        gpu::{
            create_and_write_storage_buffer, create_bind_group, create_bind_group_layout,
            create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter,
            get_device, read_from_gpu,
        },
        msm::{PARAMS, WORD_SIZE},
        shader_manager::ShaderManager,
        utils::{field_to_u8_vec_montgomery_for_gpu, u8s_to_field_without_assertion, u8s_to_fields_without_assertion},
    },
    utils::montgomery::field_to_bytes_montgomery,
};

pub async fn field_add<F: PrimeField>(a: F, b: F) -> F {
    let a_bytes = field_to_u8_vec_montgomery_for_gpu(&a, PARAMS.num_words, WORD_SIZE);
    let b_bytes = field_to_u8_vec_montgomery_for_gpu(&b, PARAMS.num_words, WORD_SIZE);

    let input_size = 1;
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 2u32.pow(chunk_size as u32) as usize;
    let num_rows = (input_size + num_columns - 1) / num_columns;
    let num_subtasks = (256 + chunk_size - 1) / chunk_size;
    let num_words = PARAMS.num_words;
    println!("Input size: {}", input_size);
    println!("Chunk size: {}", chunk_size);
    println!("Num columns: {}", num_columns);
    println!("Num rows: {}", num_rows);
    println!("Num subtasks: {}", num_subtasks);
    println!("Num words: {}", num_words);
    println!("Word size: {}", WORD_SIZE);
    println!("Params: {:?}", PARAMS);
    println!("A: {:?}", a);
    println!("B: {:?}", b);
    println!("A bytes: {:?}", a_bytes);
    println!("B bytes: {:?}", b_bytes);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size);

    let adapter = get_adapter().await;
    let (device, queue) = get_device(&adapter).await;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Field Encoder"),
    });

    let shader_code = shader_manager.gen_test_field_shader();
    let a_sb = create_and_write_storage_buffer(Some("A buffer"), &device, &a_bytes);
    let b_sb = create_and_write_storage_buffer(Some("B buffer"), &device, &b_bytes);

    let result_sb = create_storage_buffer(
        Some("Result buffer"),
        &device,
        128, // 128?
    );

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        &device,
        vec![],
        vec![&a_sb, &b_sb, &result_sb],
        vec![],
    );

    println!("Decompose bind group layout: {:?}", bind_group_layout);

    let bind_group = create_bind_group(
        Some("Bind group"),
        &device,
        &bind_group_layout,
        vec![&a_sb, &b_sb, &result_sb],
    );

    println!("Decompose bind group: {:?}", bind_group);

    let compute_pipeline = create_compute_pipeline(
        Some("Field add shader"),
        &device,
        &bind_group_layout,
        &shader_code,
        "test_field_add",
    )
    .await;

    execute_pipeline(&mut encoder, compute_pipeline, bind_group, 1, 1, 1).await;

    // Map results back from GPU to CPU.
    let data = read_from_gpu(&device, &queue, encoder, vec![result_sb], 0);

    // Destroy the GPU device object.
    device.destroy();
    println!("Data: {:?}", data);

    u8s_to_field_without_assertion(&data[0], num_words, WORD_SIZE)
}

pub fn run_webgpu_field_add<F: PrimeField>(a: F, b: F) -> F {
    pollster::block_on(run_webgpu_field_add_async(a, b))
}

pub async fn run_webgpu_field_add_async<F: PrimeField>(a: F, b: F) -> F {
    let now = Instant::now();
    let result = field_add::<F>(a, b).await;
    println!("Field add time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use halo2curves::bn256::Fr;
    use rand::thread_rng;
    use super::*;

    #[test]
    fn test_webgpu_field_add() {
        let mut rng = thread_rng();
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);

        let fast = a + b;

        let result = run_webgpu_field_add::<Fr>(a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }
}