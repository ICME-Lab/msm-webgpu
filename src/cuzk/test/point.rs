use std::time::Instant;

use halo2curves::{CurveAffine, CurveExt};
use wgpu::CommandEncoderDescriptor;

use crate::cuzk::{
    gpu::{
        create_and_write_storage_buffer, create_bind_group, create_bind_group_layout,
        create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter, get_device,
        read_from_gpu, read_from_gpu_test,
    },
    msm::{PARAMS, WORD_SIZE},
    shader_manager::ShaderManager,
    utils::{points_to_bytes_for_gpu, u8s_to_fields_without_assertion},
};

pub async fn point_op<C: CurveAffine>(op: &str, a: C, b: C) -> C::Curve {
    let a_bytes = points_to_bytes_for_gpu(&vec![a], PARAMS.num_words, WORD_SIZE);
    let b_bytes = points_to_bytes_for_gpu(&vec![b], PARAMS.num_words, WORD_SIZE);

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
        label: Some("Point Encoder"),
    });

    let shader_code = shader_manager.gen_test_point_shader();

    let a_sb = create_and_write_storage_buffer(Some("A buffer"), &device, &a_bytes);
    let b_sb = create_and_write_storage_buffer(Some("B buffer"), &device, &b_bytes);

    let result_sb = create_storage_buffer(Some("Result buffer"), &device, 240);

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

    let result = u8s_to_fields_without_assertion::<<<C as CurveAffine>::CurveExt as CurveExt>::Base>(&data[0], num_words, WORD_SIZE);

    println!("Result: {:?}", result);
    C::Curve::new_jacobian(result[0].clone(), result[1].clone(), result[2].clone()).unwrap()
}

pub fn run_webgpu_point_op<C: CurveAffine>(op: &str, a: C, b: C) -> C::Curve {
    pollster::block_on(run_webgpu_point_op_async(op, a, b))
}

pub async fn run_webgpu_point_op_async<C: CurveAffine>(op: &str, a: C, b: C) -> C::Curve {
    let now = Instant::now();
    let result = point_op::<C>(op, a, b).await;
    println!("Point op time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use group::cofactor::CofactorCurveAffine;
    use halo2curves::bn256::G1Affine;
    use rand::thread_rng;

    #[test]
    fn test_webgpu_point_add() {
        let mut rng = thread_rng();
        let a = G1Affine::random(&mut rng);
        println!("a: {:?}", a);
        let b = G1Affine::random(&mut rng);
        println!("b: {:?}", b);

        let fast = a + b;

        let result = run_webgpu_point_op::<G1Affine>("test_point_add", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }

    #[test]
    fn test_webgpu_point_add_identity() {
        let mut rng = thread_rng();
        let a = G1Affine::random(&mut rng);
        println!("a: {:?}", a);
        let b = G1Affine::identity();
        println!("b: {:?}", b);

        let fast = a + b;

        let result = run_webgpu_point_op::<G1Affine>("test_point_add", a, b);

        println!("Result: {:?}", result);
        assert_eq!(fast, result);
    }
}
