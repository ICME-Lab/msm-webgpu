use std::time::Instant;

use halo2curves::CurveAffine;
use wgpu::CommandEncoderDescriptor;

use crate::cuzk::{
    gpu::{get_adapter, get_device, read_from_gpu_test},
    msm::{PARAMS, WORD_SIZE, convert_point_coords_and_decompose_shaders, transpose_gpu},
    shader_manager::ShaderManager,
    utils::debug,
};
use crate::{points_to_bytes, scalars_to_bytes};

async fn transpose_shader<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<i32>, Vec<i32>) {
    let input_size = scalars.len();
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 1 << chunk_size;
    let num_rows = input_size.div_ceil(num_columns);
    let num_subtasks = 256_usize.div_ceil(chunk_size);
    let num_words = PARAMS.num_words;
    debug(&format!("Input size: {input_size}"));
    debug(&format!("Chunk size: {chunk_size}"));
    debug(&format!("Num columns: {num_columns}"));
    debug(&format!("Num rows: {num_rows}"));
    debug(&format!("Num subtasks: {num_subtasks}"));
    debug(&format!("Num words: {num_words}"));
    debug(&format!("Word size: {WORD_SIZE}"));
    println!("Params: {PARAMS:?}");

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
        c_num_y_workgroups,
        num_subtasks,
        num_columns,
    );

    let (_point_x_sb, _point_y_sb, scalar_chunks_sb) = convert_point_coords_and_decompose_shaders(
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

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 2. Sparse Matrix Transposition                                                         /
    //                                                                                        /
    // Compute the indices of the points which share the same                                 /
    // scalar chunks, enabling the parallel accumulation of points                            /
    // into buckets. Transposing each subtask (CSR sparse matrix)                             /
    // is a serial computation.                                                               /
    //                                                                                        /
    // The transpose step generates the CSR sparse matrix and                                 /
    // transpoes the matrix simultaneously, resulting in a                                    /
    // wide and flat matrix where width of the matrix (n) = 2 ^ chunk_size                    /
    // and height of the matrix (m) = 1.                                                      /
    ////////////////////////////////////////////////////////////////////////////////////////////

    let t_num_x_workgroups = 1;
    let t_num_y_workgroups = 1;
    let t_num_z_workgroups = 1;

    let t_shader = shader_manager.gen_transpose_shader(num_subtasks);

    let (all_csc_col_ptr_sb, all_csc_val_idxs_sb) = transpose_gpu(
        &t_shader,
        &device,
        &queue,
        &mut encoder,
        t_num_x_workgroups,
        t_num_y_workgroups,
        t_num_z_workgroups,
        input_size,
        num_columns,
        num_rows,
        num_subtasks,
        scalar_chunks_sb,
    )
    .await;

    // Map results back from GPU to CPU.
    let data = read_from_gpu_test(
        &device,
        &queue,
        encoder,
        vec![all_csc_col_ptr_sb, all_csc_val_idxs_sb],
    )
    .await;

    // Destroy the GPU device object.
    device.destroy();

    let all_csc_col_ptr = bytemuck::cast_slice::<u8, i32>(&data[0]);
    let all_csc_val_idxs = bytemuck::cast_slice::<u8, i32>(&data[1]);

    (all_csc_col_ptr.to_vec(), all_csc_val_idxs.to_vec())
}

/// Run WebGPU transpose shader sync
pub fn run_webgpu_transpose_shader<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<i32>, Vec<i32>) {
    pollster::block_on(run_webgpu_transpose_shader_async(points, scalars))
}

/// Run WebGPU transpose shader async
pub async fn run_webgpu_transpose_shader_async<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<i32>, Vec<i32>) {
    let now = Instant::now();
    let result = transpose_shader::<C>(points, scalars).await;
    println!("Transpose time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use crate::cuzk::test::cuzk::{cpu_transpose, decompose_scalars_signed};
    use crate::{sample_points, sample_scalars};

    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};

    #[test]
    fn test_webgpu_transpose_shader() {
        let input_size = 1 << 16;
        let scalars = sample_scalars::<Fr>(input_size);
        let points = sample_points::<G1Affine>(input_size);

        let chunk_size = if input_size >= 65536 { 16 } else { 4 };
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;

        let (all_csc_col_ptr, all_csc_val_idxs) =
            run_webgpu_transpose_shader::<G1Affine>(&points, &scalars);

        let decomposed_scalars = decompose_scalars_signed(&scalars, num_subtasks, chunk_size);

        // Perform multiple transpositions "in parallel"}
        let (all_csc_col_ptr_cpu, _, all_csc_val_idxs_cpu) = cpu_transpose(
            decomposed_scalars.concat(),
            num_columns,
            num_rows,
            num_subtasks,
            input_size,
        );

        assert_eq!(all_csc_col_ptr, all_csc_col_ptr_cpu);
        assert_eq!(all_csc_val_idxs, all_csc_val_idxs_cpu);
    }
}
