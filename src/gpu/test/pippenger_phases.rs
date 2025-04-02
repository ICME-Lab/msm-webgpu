use crate::gpu::*;
use crate::gpu::{run_webgpu, setup_webgpu};
use crate::halo2curves::utils::{cast_u8_to_u16, cast_u8_to_u32};
use wgpu::util::DeviceExt;


pub const WORKGROUP_SIZE: usize = 64;
pub const MAX_NUM_INVOCATIONS: usize = 2048;

pub async fn run_bucket_accumulation(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let msm_len = scalars_bytes.len() / 64;
    println!("msm_len: {:?}", msm_len);
    let num_invocations = (msm_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    println!("num_invocations: {:?}", num_invocations);
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
    let result_buffer_size = (32 * 256 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress;
    let buckets = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buckets Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let windows = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Windows Buffer"),
        size: (32 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());

    // Create a separate buffer for reading results back
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&buckets, 0, &readback_buffer, 0, result_buffer_size);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer,
            scalars_buffer,
            buckets.clone(),
            windows,
        ],
        vec![
            msm_len_buffer,
        ],
        vec![
            ("test_bucket_accumulation".to_string(), num_invocations as u32),
        ], 
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


pub async fn run_bucket_reduction(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let msm_len = scalars_bytes.len() / 64;
    println!("msm_len: {:?}", msm_len);
    let num_invocations = (msm_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    println!("num_invocations: {:?}", num_invocations);
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
    let buckets = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buckets Buffer"),
        size: (32 * 256 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let result_buffer_size = (32 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress;

    // TODO: Use create_buffer_init
    let windows = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Windows Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());

    // Create a separate buffer for reading results back
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&windows, 0, &readback_buffer, 0, result_buffer_size);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer,
            scalars_buffer,
            buckets,
            windows.clone(),
        ],
        vec![
            msm_len_buffer,
        ],
        vec![
            ("test_bucket_reduction".to_string(), num_invocations as u32),
        ], 
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