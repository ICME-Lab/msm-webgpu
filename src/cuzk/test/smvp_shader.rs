use std::{iter::zip, time::Instant};

use halo2curves::{CurveAffine, CurveExt};
use group::Curve;
use wgpu::CommandEncoderDescriptor;

use crate::{cuzk::{
    gpu::{create_storage_buffer, get_adapter, get_device, read_from_gpu_test},
    lib::{points_to_bytes, scalars_to_bytes},
    msm::{convert_point_coords_and_decompose_shaders, smvp_gpu, transpose_gpu, P, PARAMS, WORD_SIZE},
    shader_manager::ShaderManager,
    utils::{
        debug, field_to_u8_vec_montgomery_for_gpu, to_biguint_le, u8s_to_field_without_assertion, u8s_to_fields_without_assertion
    },
}, halo2curves::utils::bytes_to_field};

pub async fn smvp_shader<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> Vec<Vec<u8>> {
    let input_size = scalars.len();
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 1 << chunk_size;
    let num_rows = (input_size + num_columns - 1) / num_columns;
    let num_subtasks = (256 + chunk_size - 1) / chunk_size;
    let num_words = PARAMS.num_words;
    debug(&format!("Input size: {}", input_size));
    debug(&format!("Chunk size: {}", chunk_size));
    debug(&format!("Num columns: {}", num_columns));
    debug(&format!("Num rows: {}", num_rows));
    debug(&format!("Num subtasks: {}", num_subtasks));
    debug(&format!("Num words: {}", num_words));
    debug(&format!("Word size: {}", WORD_SIZE));
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
        c_num_y_workgroups,
        num_subtasks,
        num_columns,
    );

    // println!("C shader: {}", c_shader);

    let (point_x_sb, point_y_sb, scalar_chunks_sb) =
        convert_point_coords_and_decompose_shaders(
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

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 3. Sparse Matrix Vector Product (SMVP)                                                 /
    //                                                                                        /
    // Each thread handles accumulating points in a single bucket.                            /
    // The workgroup size and number of workgroups are designed around                        /
    // minimizing shader invocations.                                                         /
    ////////////////////////////////////////////////////////////////////////////////////////////

    let half_num_columns = num_columns / 2;
    let mut s_workgroup_size = 256;
    let mut s_num_x_workgroups = 64;
    let mut s_num_y_workgroups = (half_num_columns / s_workgroup_size) / s_num_x_workgroups;
    let mut s_num_z_workgroups = num_subtasks;

    if half_num_columns < 32768 {
        s_workgroup_size = 32;
        s_num_x_workgroups = 1;
        s_num_y_workgroups =
            ((half_num_columns / s_workgroup_size) + s_num_x_workgroups - 1) / s_num_x_workgroups;
    }

    if num_columns < 256 {
        s_workgroup_size = 1;
        s_num_x_workgroups = half_num_columns;
        s_num_y_workgroups = 1;
        s_num_z_workgroups = 1;
    }

    // This is a dynamic variable that determines the number of CSR
    // matrices processed per invocation of the shader. A safe default is 1.
    let num_subtask_chunk_size = 4;

    // Buffers that store the SMVP result, ie. bucket sums. They are
    // overwritten per iteration.
    let bucket_sum_coord_bytelength = (num_columns / 2) * num_words * 4 * num_subtasks;
    let bucket_sum_x_sb = create_storage_buffer(
        Some("Bucket sum X buffer"),
        &device,
        bucket_sum_coord_bytelength as u64,
    );
    let bucket_sum_y_sb = create_storage_buffer(
        Some("Bucket sum Y buffer"),
        &device,
        bucket_sum_coord_bytelength as u64,
    );
    let bucket_sum_z_sb = create_storage_buffer(
        Some("Bucket sum Z buffer"),
        &device,
        bucket_sum_coord_bytelength as u64,
    );
    let smvp_shader = shader_manager.gen_smvp_shader(s_workgroup_size, num_columns);

    for offset in (0..num_subtasks).step_by(num_subtask_chunk_size) {
        smvp_gpu(
            &smvp_shader,
            s_num_x_workgroups / (num_subtasks / num_subtask_chunk_size),
            s_num_y_workgroups,
            s_num_z_workgroups,
            offset,
            &device,
            &queue,
            &mut encoder,
            input_size,
            &all_csc_col_ptr_sb,
            &point_x_sb,
            &point_y_sb,
            &all_csc_val_idxs_sb,
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
        )
        .await;
    }
    // Map results back from GPU to CPU.
    let data = read_from_gpu_test(
        &device,
        &queue,
        encoder,
        vec![bucket_sum_x_sb, bucket_sum_y_sb, bucket_sum_z_sb],
    )
    .await;

    // Destroy the GPU device object.
    device.destroy();

    let p_x = bytemuck::cast_slice::<u8, u32>(&data[0]).chunks(20);
    let p_y = bytemuck::cast_slice::<u8, u32>(&data[1]).chunks(20);
    let p_z = bytemuck::cast_slice::<u8, u32>(&data[2]).chunks(20);

    let p = zip(zip(p_x, p_y), p_z)
        .enumerate()
        .map(|(i, ((x, y), z))| {
            let p_x_biguint_montgomery = to_biguint_le(&x.to_vec(), num_words, WORD_SIZE as u32);
            let p_y_biguint_montgomery = to_biguint_le(&y.to_vec(), num_words, WORD_SIZE as u32);
            let p_z_biguint_montgomery = to_biguint_le(&z.to_vec(), num_words, WORD_SIZE as u32);
        
            let p_x_biguint = p_x_biguint_montgomery * &PARAMS.rinv % P.clone();
            let p_y_biguint = p_y_biguint_montgomery * &PARAMS.rinv % P.clone();
            let p_z_biguint = p_z_biguint_montgomery * &PARAMS.rinv % P.clone();
            let p_x_field = bytes_to_field(&p_x_biguint.to_bytes_le());
            let p_y_field = bytes_to_field(&p_y_biguint.to_bytes_le());
            let p_z_field = bytes_to_field(&p_z_biguint.to_bytes_le());
            let p = C::Curve::new_jacobian(p_x_field, p_y_field, p_z_field).unwrap();
            p.to_affine()
        })
        .collect::<Vec<_>>();
    data
}

pub fn run_webgpu_smvp_shader<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> Vec<Vec<u8>> {
    pollster::block_on(run_webgpu_smvp_shader_async(points, scalars))
}

pub async fn run_webgpu_smvp_shader_async<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> Vec<Vec<u8>> {
    let now = Instant::now();
    let result = smvp_shader::<C>(points, scalars).await;
    println!("SMVP time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use crate::cuzk::{
        lib::{sample_points, sample_scalars},
        test::cuzk::decompose_scalars_signed,
    };

    use super::*;
    use halo2curves::bn256::{Fr, G1Affine};

    #[test]
    fn test_webgpu_smvp_shader() {
        let input_size = 1 << 16;
        let scalars = sample_scalars::<Fr>(input_size);
        let points = sample_points::<G1Affine>(input_size);

        let chunk_size: usize = 16;
        let num_columns = 1 << chunk_size;
        let num_rows = (input_size + num_columns - 1) / num_columns;
        let num_chunks_per_scalar = (256 + chunk_size - 1) / chunk_size;
        let num_subtasks = num_chunks_per_scalar;

        let result_bucket_sums = run_webgpu_smvp_shader::<G1Affine>(&points, &scalars);
        // println!("Result bucket sums: {:?}", result_bucket_sums);
    }
}
