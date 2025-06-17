use std::{iter::zip, time::Instant};

use halo2curves::CurveAffine;
use wgpu::CommandEncoderDescriptor;

use msm_webgpu::cuzk::{
    gpu::{get_adapter, get_device, read_from_gpu_test},
    msm::{convert_point_coords_and_decompose_shaders, WORD_SIZE},
    shader_manager::ShaderManager,
    utils::{bytes_to_field, compute_misc_params, compute_p, debug, to_biguint_le},
};
use msm_webgpu::{points_to_bytes, scalars_to_bytes};

async fn decompose_shader<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<C>, Vec<u8>) {
    let p = compute_p::<C>();
    let params = compute_misc_params(&p, WORD_SIZE);
    let input_size = scalars.len();
    let chunk_size = if input_size >= 65536 { 16 } else { 4 };
    let num_columns = 1 << chunk_size;
    let num_rows = input_size.div_ceil(num_columns);
    let num_subtasks = 256_usize.div_ceil(chunk_size);
    let num_words = params.num_words;
    debug(&format!("Input size: {input_size}"));
    debug(&format!("Chunk size: {chunk_size}"));
    debug(&format!("Num columns: {num_columns}"));
    debug(&format!("Num rows: {num_rows}"));
    debug(&format!("Num subtasks: {num_subtasks}"));
    debug(&format!("Num words: {num_words}"));
    debug(&format!("Word size: {WORD_SIZE}"));
    debug(&format!("Params: {params:?}"));

    let point_bytes = points_to_bytes(points);
    let scalar_bytes = scalars_to_bytes(scalars);

    let shader_manager = ShaderManager::new(WORD_SIZE, chunk_size, input_size, &params);

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
    let data = read_from_gpu_test(
        &device,
        &queue,
        encoder,
        vec![point_x_sb, point_y_sb, scalar_chunks_sb],
    )
    .await;

    // Destroy the GPU device object.
    device.destroy();

    let p_x = bytemuck::cast_slice::<u8, u32>(&data[0]).chunks(20);
    let p_y = bytemuck::cast_slice::<u8, u32>(&data[1]).chunks(20);

    let p = zip(p_x, p_y)
        .map(|(x, y)| {
            let p_x_biguint_montgomery = to_biguint_le(x, num_words, WORD_SIZE as u32);
            let p_y_biguint_montgomery = to_biguint_le(y, num_words, WORD_SIZE as u32);

            let p_x_biguint = p_x_biguint_montgomery * &params.rinv % p.clone();
            let p_y_biguint = p_y_biguint_montgomery * &params.rinv % p.clone();
            let p_x_field = bytes_to_field(&p_x_biguint.to_bytes_le());
            let p_y_field = bytes_to_field(&p_y_biguint.to_bytes_le());

            let p = C::from_xy(p_x_field, p_y_field);
            p.unwrap()
        })
        .collect::<Vec<_>>();
    (p, data[2].clone())
}

/// Run WebGPU decompose sync
pub fn run_webgpu_decompose<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<C>, Vec<u8>) {
    pollster::block_on(run_webgpu_decompose_async(points, scalars))
}

/// Run WebGPU decompose async
pub async fn run_webgpu_decompose_async<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
) -> (Vec<C>, Vec<u8>) {
    let now = Instant::now();
    let result = decompose_shader::<C>(points, scalars).await;
    println!("Decompose time: {:?}", now.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use msm_webgpu::{sample_points, sample_scalars};

    use halo2curves::bn256::{Fr, G1Affine};
    use halo2curves::pasta::pallas::{Affine as PallasAffine, Scalar as PallasScalar};
    #[test]
    fn test_decompose_bn256() {
        let input_size = 1 << 16;
        let scalars = sample_scalars::<Fr>(input_size);
        let points = sample_points::<G1Affine>(input_size);

        let (result_points, _result_scalars) = run_webgpu_decompose::<G1Affine>(&points, &scalars);
        assert_eq!(result_points, points);
    }

    #[test]
    fn test_decompose_pallas() {
        let input_size = 1 << 16;
        let scalars = sample_scalars::<PallasScalar>(input_size);
        let points = sample_points::<PallasAffine>(input_size);

        let (result_points, _result_scalars) = run_webgpu_decompose::<PallasAffine>(&points, &scalars);
        assert_eq!(result_points, points);
    }
}
