use crate::gpu::*;
use crate::gpu::{run_webgpu, setup_webgpu};
use crate::halo2curves::utils::cast_u8_to_u16;
use wgpu::util::DeviceExt;

pub async fn run_msm(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MSM Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });
    let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Points Buffer"),
        contents: points_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let scalars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scalars Buffer"),
        contents: scalars_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    // The result buffer must be large enough to hold final data
    let result_buffer_size = (NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mem = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Memory Buffer"),
        size: (MSM_SIZE * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let pippenger_pow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pippenger Powerset Sums Buffer"),
        size: (256 * NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let pippenger_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pippenger Sum Buffer"),
        size: (256 * NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let msm_len_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MSM Length Buffer"),
        contents: &((scalars_bytes.len() / 64) as u32).to_le_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create a separate buffer for reading results back
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MSM Compute Pipeline (main)"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_buffer_size);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer,
            scalars_buffer,
            result_buffer.clone(),
            mem,
            pippenger_pow_buffer,
            pippenger_sum_buffer,
            msm_len_buffer,
        ],
        vec!["main".to_string()], // TODO: add aggregate
        compute_pipeline_fn,
        readback_buffer.clone(),
        copy_results_to_encoder,
    )
    .await;

    let buffer_slice = result.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();

    let output_u16 = cast_u8_to_u16(&data);
    drop(data);
    result.unmap();

    output_u16
}
