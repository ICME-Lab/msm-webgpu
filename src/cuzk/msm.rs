use group::Group;
use halo2curves::CurveAffine;
use halo2curves::CurveExt;
use num_bigint::BigUint;
use num_traits::Num;
use once_cell::sync::Lazy;
use wgpu::{Buffer, CommandEncoder, CommandEncoderDescriptor, Device, Queue};

use crate::cuzk::gpu::{
    create_and_write_storage_buffer, create_and_write_uniform_buffer, create_bind_group,
    create_bind_group_layout, create_compute_pipeline, create_storage_buffer, execute_pipeline,
    get_adapter, get_device, read_from_gpu,
};
use crate::cuzk::shader_manager::ShaderManager;
use crate::cuzk::utils::compute_p;
use crate::cuzk::utils::to_biguint_le;
use crate::{points_to_bytes, scalars_to_bytes};

use super::utils::bytes_to_field;
use super::utils::{MiscParams, compute_misc_params};
use ff::{Field, PrimeField};

/// 13-bit limbs.
pub const WORD_SIZE: usize = 13;

fn pad_scalars<C: CurveAffine>(scalars: &[C::Scalar]) -> Vec<C::Scalar> {
    let n = scalars.len();
    let l = n.next_power_of_two();
    let diff = l - n;
    let mut padded_scalars = vec![C::Scalar::ZERO; l];
    padded_scalars[..scalars.len()].copy_from_slice(scalars);
    let mut rng = rand::thread_rng();
    for i in 0..(diff / 2) {
        let a = C::Scalar::random(&mut rng);
        padded_scalars[n + i] = a;
        padded_scalars[n + diff / 2 + i] = -a;
    }
    padded_scalars
}

fn pad_points<C: CurveAffine>(points: &[C]) -> Vec<C> {
    let n = points.len();
    let l = n.next_power_of_two();
    let mut padded_points = vec![C::generator(); l];
    padded_points[..points.len()].copy_from_slice(points);
    padded_points
}

/*
 * End-to-end implementation of the modified cuZK MSM algorithm by Lu et al,
 * 2022: https://eprint.iacr.org/2022/1321.pdf
 */
pub async fn compute_msm<C: CurveAffine>(points: &[C], scalars: &[C::Scalar]) -> C::Curve {
    let p = compute_p::<C>();
    let params = compute_misc_params(&p, WORD_SIZE);
    let padded_scalars = pad_scalars::<C>(scalars);
    let padded_points = pad_points::<C>(points);
    let input_size = padded_scalars.len();
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 1 << chunk_size;
    let num_rows = input_size.div_ceil(num_columns);
    let num_subtasks = 256_usize.div_ceil(chunk_size);
    let num_words = params.num_words;

    let point_bytes = points_to_bytes(&padded_points);
    let scalar_bytes = scalars_to_bytes(&padded_scalars);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size, &params);

    let adapter = get_adapter().await;
    let (device, queue) = get_device(&adapter).await;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("MSM Encoder"),
    });

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Decompose scalars into chunk_size windows using signed bucket indices.             /
    ////////////////////////////////////////////////////////////////////////////////////////////

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
    } else if input_size > 32768 && input_size <= 131072 {
        c_workgroup_size = 256;
        c_num_x_workgroups = 8;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if input_size > 131072 && input_size <= 1048576 {
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
            (half_num_columns / s_workgroup_size).div_ceil(s_num_x_workgroups);
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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // 4. Bucket Reduction                                                                     /
    //                                                                                         /
    // Performs a parallelized running-sum by computing a serieds of point additions,          /
    // followed by a scalar multiplication (Algorithm 4 of the cuZK paper).                    /
    /////////////////////////////////////////////////////////////////////////////////////////////

    // This is a dynamic variable that determines the number of CSR
    // matrices processed per invocation of the BPR shader. A safe default is 1.
    let num_subtasks_per_bpr_1 = 16;

    let b_num_x_workgroups = num_subtasks_per_bpr_1;
    let b_num_y_workgroups = 1;
    let b_num_z_workgroups = 1;
    let b_workgroup_size = 256;

    // Buffers that store the bucket points reduction (BPR) output.
    let g_points_coord_bytelength = num_subtasks * b_workgroup_size * num_words * 4;
    let g_points_x_sb = create_storage_buffer(
        Some("Bucket points reduction X buffer"),
        &device,
        g_points_coord_bytelength as u64,
    );
    let g_points_y_sb = create_storage_buffer(
        Some("Bucket points reduction Y buffer"),
        &device,
        g_points_coord_bytelength as u64,
    );
    let g_points_z_sb = create_storage_buffer(
        Some("Bucket points reduction Z buffer"),
        &device,
        g_points_coord_bytelength as u64,
    );

    let bpr_shader = shader_manager.gen_bpr_shader(b_workgroup_size);

    // Stage 1: Bucket points reduction (BPR)
    for subtask_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_1) {
        bpr_1(
            &bpr_shader,
            subtask_idx,
            b_num_x_workgroups,
            b_num_y_workgroups,
            b_num_z_workgroups,
            num_columns,
            &device,
            &queue,
            &mut encoder,
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
        )
        .await;
    }

    let num_subtasks_per_bpr_2 = 16;
    let b_2_num_x_workgroups = num_subtasks_per_bpr_2;

    // Stage 2: Bucket points reduction (BPR).
    for subtask_idx in (0..num_subtasks).step_by(num_subtasks_per_bpr_2) {
        bpr_2(
            &bpr_shader,
            subtask_idx,
            b_2_num_x_workgroups,
            1,
            1,
            num_columns,
            &device,
            &queue,
            &mut encoder,
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
        )
        .await;
    }

    // Map results back from GPU to CPU.
    let data = read_from_gpu(
        &device,
        &queue,
        encoder,
        vec![g_points_x_sb, g_points_y_sb, g_points_z_sb],
    )
    .await;

    // Destroy the GPU device object.
    device.destroy();

    let mut points = vec![];
    let r_inv = params.clone().rinv;

    let g_points_x = bytemuck::cast_slice::<u8, u32>(&data[0])
        .chunks(num_words)
        .map(|x| {
            let x_biguint_montgomery = to_biguint_le(x, num_words, WORD_SIZE as u32);
            let x_biguint = x_biguint_montgomery * &r_inv % p.clone();
            
            bytes_to_field(&x_biguint.to_bytes_le())
        })
        .collect::<Vec<_>>();
    let g_points_y = bytemuck::cast_slice::<u8, u32>(&data[1])
        .chunks(num_words)
        .map(|y| {
            let y_biguint_montgomery = to_biguint_le(y, num_words, WORD_SIZE as u32);
            let y_biguint = y_biguint_montgomery * &r_inv % p.clone();
            
            bytes_to_field(&y_biguint.to_bytes_le())
        })
        .collect::<Vec<_>>();
    let g_points_z = bytemuck::cast_slice::<u8, u32>(&data[2])
        .chunks(num_words)
        .map(|z| {
            let z_biguint_montgomery = to_biguint_le(z, num_words, WORD_SIZE as u32);
            let z_biguint = z_biguint_montgomery * &r_inv % p.clone();
            
            bytes_to_field(&z_biguint.to_bytes_le())
        })
        .collect::<Vec<_>>();

    // TODO: Use from_montgomery_repr passing a valid R^2 as a parameter for performance
    // let g_points_x = data[0]
    //     .chunks(num_words * 4)
    //     .map(|x| {
    //         let x_field = u8s_to_field_without_assertion(&x, num_words, WORD_SIZE);
    //         x_field
    //     })
    //     .collect::<Vec<_>>();

    for i in 0..num_subtasks {
        let mut point = C::Curve::identity();
        for j in 0..b_workgroup_size {
            let reduced_point = C::Curve::new_jacobian(
                g_points_x[i * b_workgroup_size + j],
                g_points_y[i * b_workgroup_size + j],
                g_points_z[i * b_workgroup_size + j],
            )
            .unwrap();
            point += reduced_point;
        }
        points.push(point);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 5. Horner's Method                                                                     /
    //                                                                                        /
    // Calculate the final result using Horner's method (Formula 3 of the cuZK paper)         /
    ////////////////////////////////////////////////////////////////////////////////////////////

    let m = C::ScalarExt::from(1 << chunk_size);
    let mut result = points[points.len() - 1];
    for i in (0..points.len() - 1).rev() {
        result = result * m + points[i];
    }
    result
}

/****************************************************** WGSL Shader Invocations ******************************************************/

/*
 * Prepares and executes the shader for decomposing scalars into chunk_size
 * windows using the signed bucket index technique.
 *
 * ASSUMPTION: the vast majority of WebGPU-enabled consumer devices have a
 * maximum buffer size of at least 268435456 bytes.
 *
 * The default maximum buffer size is 268435456 bytes. Since each point
 * consumes 320 bytes, a maximum of around 2 ** 19 points can be stored in a
 * single buffer. If, however, we use 2 buffers - one for each point coordinate
 * X and Y - we can support larger input sizes.
 * Our implementation, however, will only support up to 2 ** 20 points.
 *
 * Furthremore, there is a limit of 8 storage buffers per shader. As such, we
 * do not calculate the T and Z coordinates in this shader. Rather, we do so in
 * the SMVP shader.
 *
 */

/// Convert point coordinates and decompose shaders
pub async fn convert_point_coords_and_decompose_shaders(
    shader_code: &str,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    points_bytes: &[u8],
    scalars_bytes: &[u8],
    num_subtasks: usize,
    chunk_size: usize,
    num_words: usize,
) -> (Buffer, Buffer, Buffer) {
    assert!(num_subtasks * chunk_size == 256);
    let input_size = scalars_bytes.len() / 32;
    let points_sb = create_and_write_storage_buffer(Some("Points buffer"), device, points_bytes);
    let scalars_sb = create_and_write_storage_buffer(Some("Scalars buffer"), device, scalars_bytes);

    let points_x_sb = create_storage_buffer(
        Some("Point X buffer"),
        device,
        (input_size * num_words * 4) as u64,
    );
    let points_y_sb = create_storage_buffer(
        Some("Point Y buffer"),
        device,
        (input_size * num_words * 4) as u64,
    );
    // Output storage buffers.
    let scalar_chunks_sb = create_storage_buffer(
        Some("Scalar chunks buffer"),
        device,
        (input_size * num_subtasks * 4) as u64, // TODO: Check this
    );

    // Uniform storage buffer.
    let params_bytes = to_u8s_for_gpu([input_size].to_vec());
    let params_ub =
        create_and_write_uniform_buffer(Some("Params buffer"), device, queue, &params_bytes);

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        device,
        vec![&points_sb, &scalars_sb],
        vec![&points_x_sb, &points_y_sb, &scalar_chunks_sb],
        vec![&params_ub],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        device,
        &bind_group_layout,
        vec![
            &points_sb,
            &scalars_sb,
            &points_x_sb,
            &points_y_sb,
            &scalar_chunks_sb,
            &params_ub,
        ],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Convert point coords and decompose shader"),
        device,
        &bind_group_layout,
        shader_code,
        "main",
    )
    .await;

    execute_pipeline(
        encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    )
    .await;

    (points_x_sb, points_y_sb, scalar_chunks_sb)
}

/*
 * Perform a modified version of CSR matrix transposition, which comes before
 * SMVP. Essentially, this step generates the point indices for each thread in
 * the SMVP step which corresponds to a particular bucket.
 */
pub async fn transpose_gpu(
    shader_code: &str,
    device: &Device,
    queue: &Queue,
    command_encoder: &mut CommandEncoder,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    input_size: usize,
    num_columns: usize,
    num_rows: usize,
    num_subtasks: usize,
    scalar_chunks_sb: Buffer,
) -> (Buffer, Buffer) {
    // Input storage buffers.
    let all_csc_col_ptr_sb = create_storage_buffer(
        Some("All CSC col"),
        device,
        (num_subtasks * (num_columns + 1) * 4) as u64,
    );
    let all_csc_val_idxs_sb =
        create_storage_buffer(Some("All CSC Val Indexes"), device, scalar_chunks_sb.size());
    let all_curr_sb = create_storage_buffer(
        Some("All Current"),
        device,
        (num_subtasks * num_columns * 4) as u64,
    );

    // Uniform storage buffer.
    let params_bytes = to_u8s_for_gpu([num_rows, num_columns, input_size].to_vec());
    let params_ub = create_and_write_uniform_buffer(
        Some("Transpose GPU Uniform Params"),
        device,
        queue,
        &params_bytes,
    );

    let bind_group_layout = create_bind_group_layout(
        Some("Transpose GPU Bind Group Layout"),
        device,
        vec![&scalar_chunks_sb],
        vec![&all_csc_col_ptr_sb, &all_csc_val_idxs_sb, &all_curr_sb],
        vec![&params_ub],
    );

    let bind_group = create_bind_group(
        Some("Transpose GPU Bind Group"),
        device,
        &bind_group_layout,
        vec![
            &scalar_chunks_sb,
            &all_csc_col_ptr_sb,
            &all_csc_val_idxs_sb,
            &all_curr_sb,
            &params_ub,
        ],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Transpose GPU Compute Pipeline"),
        device,
        &bind_group_layout,
        shader_code,
        "main",
    )
    .await;

    execute_pipeline(
        command_encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    )
    .await;

    (all_csc_col_ptr_sb, all_csc_val_idxs_sb)
}

// TODO: Use bytemuck
pub fn to_u8s_for_gpu(vals: Vec<usize>) -> Vec<u8> {
    let max: u64 = 1 << 32;
    let mut buf = vec![];
    for val in vals {
        assert!((val as u64) < max);
        buf.extend_from_slice(&(val as u32).to_le_bytes());
    }
    buf
}

/*
 * Compute the bucket sums and perform scalar multiplication with the bucket indices.
 */
pub async fn smvp_gpu(
    shader_code: &str,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    offset: usize,
    device: &Device,
    queue: &Queue,
    command_encoder: &mut CommandEncoder,
    input_size: usize,
    all_csc_col_ptr_sb: &Buffer,
    point_x_sb: &Buffer,
    point_y_sb: &Buffer,
    all_csc_val_idxs_sb: &Buffer,
    bucket_sum_x_sb: &Buffer,
    bucket_sum_y_sb: &Buffer,
    bucket_sum_z_sb: &Buffer,
) {
    // Uniform Storage Buffer.
    let params_bytes = to_u8s_for_gpu(vec![input_size, num_y_workgroups, num_z_workgroups, offset]);
    let params_ub = create_and_write_uniform_buffer(None, device, queue, &params_bytes);

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        device,
        vec![
            &all_csc_col_ptr_sb,
            &all_csc_val_idxs_sb,
            &point_x_sb,
            &point_y_sb,
        ],
        vec![&bucket_sum_x_sb, &bucket_sum_y_sb, &bucket_sum_z_sb],
        vec![&params_ub],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        device,
        &bind_group_layout,
        vec![
            &all_csc_col_ptr_sb,
            &all_csc_val_idxs_sb,
            &point_x_sb,
            &point_y_sb,
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &params_ub,
        ],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Compute pipeline"),
        device,
        &bind_group_layout,
        shader_code,
        "main",
    )
    .await;

    execute_pipeline(
        command_encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    )
    .await;
}

/// Batch product reduction shader 1
pub async fn bpr_1(
    shader_code: &str,
    subtask_idx: usize,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    num_columns: usize,
    device: &Device,
    queue: &Queue,
    command_encoder: &mut CommandEncoder,
    bucket_sum_x_sb: &Buffer,
    bucket_sum_y_sb: &Buffer,
    bucket_sum_z_sb: &Buffer,
    g_points_x_sb: &Buffer,
    g_points_y_sb: &Buffer,
    g_points_z_sb: &Buffer,
) {
    // Uniform storage buffer.
    let params_bytes = to_u8s_for_gpu(vec![subtask_idx, num_columns, num_x_workgroups]);
    let params_ub = create_and_write_uniform_buffer(None, device, queue, &params_bytes);

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        device,
        vec![],
        vec![
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
        ],
        vec![&params_ub],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        device,
        &bind_group_layout,
        vec![
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
            &params_ub,
        ],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Compute pipeline"),
        device,
        &bind_group_layout,
        shader_code,
        "stage_1",
    )
    .await;

    execute_pipeline(
        command_encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    )
    .await;
}

/// Batch product reduction shader 2
pub async fn bpr_2(
    shader_code: &str,
    subtask_idx: usize,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    num_columns: usize,
    device: &Device,
    queue: &Queue,
    command_encoder: &mut CommandEncoder,
    bucket_sum_x_sb: &Buffer,
    bucket_sum_y_sb: &Buffer,
    bucket_sum_z_sb: &Buffer,
    g_points_x_sb: &Buffer,
    g_points_y_sb: &Buffer,
    g_points_z_sb: &Buffer,
) {
    // Uniform storage buffer.
    let params_bytes = to_u8s_for_gpu(vec![subtask_idx, num_columns, num_x_workgroups]);
    let params_ub = create_and_write_uniform_buffer(None, device, queue, &params_bytes);

    let bind_group_layout = create_bind_group_layout(
        Some("Bind group layout"),
        device,
        vec![],
        vec![
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
        ],
        vec![&params_ub],
    );

    let bind_group = create_bind_group(
        Some("Bind group"),
        device,
        &bind_group_layout,
        vec![
            &bucket_sum_x_sb,
            &bucket_sum_y_sb,
            &bucket_sum_z_sb,
            &g_points_x_sb,
            &g_points_y_sb,
            &g_points_z_sb,
            &params_ub,
        ],
    );

    let compute_pipeline = create_compute_pipeline(
        Some("Compute pipeline"),
        device,
        &bind_group_layout,
        shader_code,
        "stage_2",
    )
    .await;

    execute_pipeline(
        command_encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    )
    .await;
}
