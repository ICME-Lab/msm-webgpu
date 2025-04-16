use halo2curves::bn256::G1Affine;
use wgpu::{Buffer, CommandEncoder, CommandEncoderDescriptor, Device, Queue};

use crate::cuzk::gpu::{
    create_and_write_storage_buffer, create_and_write_uniform_buffer, create_bind_group, create_bind_group_layout, create_compute_pipeline, create_storage_buffer, execute_pipeline, get_adapter, get_device
};
use crate::cuzk::shader_manager::ShaderManager;


// TODO: HARDCODE THE VALUE FOR BN256
pub fn calc_num_words(word_size: usize) -> usize {
    let p_width = 254;
    let mut num_words = p_width / word_size;
    while num_words * word_size < p_width {
        num_words += 1;
    }
    num_words
}

/// 13-bit limbs.
const WORD_SIZE: usize = 13;

/*
 * End-to-end implementation of the modified cuZK MSM algorithm by Lu et al,
 * 2022: https://eprint.iacr.org/2022/1321.pdf
 *
 * Many aspects of cuZK were adapted and modified for our submission, and some
 * aspects were omitted. As such, please refer to the documentation
 * (https://hackmd.io/HNH0DcSqSka4hAaIfJNHEA) we have written for a more accurate
 * description of our work. We also used techniques by previous ZPrize contestations.
 * In summary, we took the following approach:
 *
 * 1. Perform as much of the computation within the GPU as possible, in order
 *    to minimse CPU-GPU and GPU-CPU data transfer, which is slow.
 * 2. Use optimizations inspired by previous years' submissions, such as:
 *    - Montgomery multiplication and Barrett reduction with 13-bit limb sizes
 *    - Signed bucket indices
 * 3. Careful memory management to stay within WebGPU's default buffer size
 *    limits.
 * 4. Perform the final computation of (Horner's rule) in the CPU instead of the GPU,
 *    as the number of points is small, and the time taken to compile a shader to
 *    perform this computation is greater than the time it takes for the CPU to do so.
 */
pub async fn compute_msm(points: &[u8], scalars: &[u8]) -> G1Affine {
    let input_size = scalars.len() / 32;
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 2u32.pow(chunk_size as u32) as usize;
    let num_rows = (input_size + num_columns - 1) / num_columns;
    let num_subtasks = (256 + chunk_size - 1) / chunk_size;
    let num_words = calc_num_words(WORD_SIZE);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size, num_words);

    let adapter = get_adapter().await;
    let (device, queue) = get_device(&adapter).await;
    let encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("MSM Encoder"),
    });

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// 1. Point Coordinate Conversation and Scalar Decomposition                              /
    ///                                                                                        /
    /// (1) Convert elliptic curve points (ETE Affine coordinates) into 13-bit limbs,          /
    /// and represented internally in Montgomery form by using Barret Reduction.               /
    ///                                                                                        /
    /// (2) Decompose scalars into chunk_size windows using signed bucket indices.             /
    ////////////////////////////////////////////////////////////////////////////////////////////

    /// Total thread count = workgroup_size * #x workgroups * #y workgroups * #z workgroups.
    let mut c_workgroup_size = 64;
    let mut c_num_x_workgroups = 128;
    let mut c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    let c_num_z_workgroups = 1;

    if (input_size <= 256) {
        c_workgroup_size = input_size;
        c_num_x_workgroups = 1;
        c_num_y_workgroups = 1;
    } else if (input_size > 256 && input_size <= 32768) {
        c_workgroup_size = 64;
        c_num_x_workgroups = 4;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if (input_size > 32768 && input_size <= 65536) {
        c_workgroup_size = 256;
        c_num_x_workgroups = 8;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if (input_size > 65536 && input_size <= 131072) {
        c_workgroup_size = 256;
        c_num_x_workgroups = 8;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if (input_size > 131072 && input_size <= 262144) {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if (input_size > 262144 && input_size <= 524288) {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    } else if (input_size > 524288 && input_size <= 1048576) {
        c_workgroup_size = 256;
        c_num_x_workgroups = 32;
        c_num_y_workgroups = input_size / c_workgroup_size / c_num_x_workgroups;
    }

    let c_shader = shader_manager.gen_decomp_scalars_shader(c_workgroup_size);

    unimplemented!()
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

pub async fn convert_point_coords_and_decompose_shaders(
    shader_code: &str,
    num_x_workgroups: usize,
    num_y_workgroups: usize,
    num_z_workgroups: usize,
    device: &Device,
    queue: &Queue,
    encoder: &mut CommandEncoder,
    points_buffer: &[u8],
    scalars_buffer: &[u8],
    num_words: usize,
    num_subtasks: usize,
    chunk_size: usize,
) -> (Buffer, Buffer, Buffer) {
    assert!(num_subtasks * chunk_size == 256);
    let input_size = scalars_buffer.len() / 32;
    let points_sb = create_and_write_storage_buffer(Some("Points buffer"), device, points_buffer);
    let scalars_sb =
        create_and_write_storage_buffer(Some("Scalars buffer"), device, scalars_buffer);

    // Output storage buffers.
    let point_x_sb = create_storage_buffer(
        Some("Point X buffer"),
        device,
        (input_size * num_words * 4) as u64,
    );
    let point_y_sb = create_storage_buffer(
        Some("Point Y buffer"),
        device,
        (input_size * num_words * 4) as u64,
    );
    let scalar_chunks_sb = create_storage_buffer(
        Some("Scalar chunks buffer"),
        device,
        (input_size * num_subtasks * 4) as u64,
    );

    // Uniform storage buffer.
    //   let params_bytes = numbers_to_u8s_for_gpu([input_size]);
    let params_ub =
        create_and_write_uniform_buffer(Some("Params buffer"), device, queue, &[input_size]);

    let bind_group_layout = create_bind_group_layout(
        device,
        Some("Bind group layout"),
        vec![points_sb, scalars_sb],
        vec![point_x_sb, point_y_sb, scalar_chunks_sb],
        vec![params_ub],
    );

    let bind_group = create_bind_group(
        device,
        Some("Bind group"),
        bind_group_layout,
        vec![points_sb, scalars_sb, point_x_sb, point_y_sb, scalar_chunks_sb],
    );

    let compute_pipeline = create_compute_pipeline(
        device,
        Some("Convert point coords and decompose shader"),
        bind_group_layout,
        shader_code,
        "main",
    ).await;

    execute_pipeline(
        &mut encoder,
        compute_pipeline,
        bind_group,
        num_x_workgroups as u32,
        num_y_workgroups as u32,
        num_z_workgroups as u32,
    );
    
    (point_x_sb, point_y_sb, scalar_chunks_sb)
}
