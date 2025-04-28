use std::{iter::zip, time::Instant};

use ff::PrimeField;
use halo2curves::{CurveAffine, CurveExt};
use wgpu::CommandEncoderDescriptor;

use crate::cuzk::{
    gpu::{
        create_and_write_storage_buffer, create_bind_group, create_bind_group_layout,
        create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter, get_device,
        read_from_gpu,
    }, lib::{points_to_bytes, scalars_to_bytes}, msm::{convert_point_coords_and_decompose_shaders, PARAMS, WORD_SIZE}, shader_manager::ShaderManager, utils::{field_to_u8_vec_montgomery_for_gpu, u8s_to_field_without_assertion, u8s_to_fields_without_assertion}
};

pub async fn decompose<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> Vec<C> {
    let input_size = scalars.len();
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 1 << chunk_size;
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

    let point_bytes = points_to_bytes(points);
    let scalar_bytes = scalars_to_bytes(scalars);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size);

    let adapter = get_adapter().await;
    let (device, queue) = get_device(&adapter).await;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Decompose Encoder"),
    });


    // Total thread count = workgroup_size * #x workgroups * #y workgroups * #z workgroups.
    let mut c_workgroup_size = 64;
    let mut c_num_x_workgroups = 128;
    let mut c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    let c_num_z_workgroups = 1;

    if input_size <= 256 {
        c_workgroup_size = input_size;
        c_num_x_workgroups = 1;
        c_num_y_workgroups = 1;
    } else if input_size > 256 && input_size <= 32768 {
        c_workgroup_size = 64;
        c_num_x_workgroups = 4;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 32768 && input_size <= 65536 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 8;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 65536 && input_size <= 131072 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 8;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 131072 && input_size <= 262144 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 262144 && input_size <= 524288 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 524288 && input_size <= 1048576 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    }

    let c_shader = shader_manager.gen_decomp_scalars_shader(
        c_workgroup_size,
        c_num_x_workgroups,
        c_num_y_workgroups,
        num_columns,
    );

    // println!("C shader: {}", c_shader);

    let (point_x_sb, point_y_sb, scalar_chunks_sb) = convert_point_coords_and_decompose_shaders(
        &c_shader,
        c_num_x_workgroups,
        c_num_y_workgroups,
        c_num_z_workgroups,
        &device,
        &queue,
        &mut encoder,
        &point_bytes,
        &scalar_bytes,
        num_subtasks,
        chunk_size,
        num_words,
    )
    .await;
    // Map results back from GPU to CPU.
    let data = read_from_gpu(&device, &queue, encoder, vec![point_x_sb, point_y_sb, scalar_chunks_sb]).await;

    // Destroy the GPU device object.
    device.destroy();

    println!("Data: {:?}", data[0]);

    let p_x = u8s_to_fields_without_assertion::<
       C::Base,
    >(&data[0], num_words, WORD_SIZE);
    let p_y = u8s_to_fields_without_assertion::<
        C::Base,
    >(&data[1], num_words, WORD_SIZE);

    zip(p_x, p_y).map(|(x, y)| C::from_xy(x, y).unwrap()).collect::<Vec<_>>()
    

}

pub fn run_webgpu_decompose<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> Vec<C> {
    pollster::block_on(run_webgpu_decompose_async(points, scalars))
}

pub async fn run_webgpu_decompose_async<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> Vec<C> {
    let now = Instant::now();
    let result = decompose::<C>(points, scalars).await;
    println!("Decompose time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use crate::cuzk::lib::{sample_points, sample_scalars};

    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};

    #[test]
    fn test_webgpu_decompose() {
        let scalars = sample_scalars::<Fr>(10);
        let points = sample_points::<G1Affine>(10);

        let result = run_webgpu_decompose::<G1Affine>(&points, &scalars);
        assert_eq!(result, points);
    }
}
