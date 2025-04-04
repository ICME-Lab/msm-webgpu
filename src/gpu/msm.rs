use crate::gpu::*;
use crate::gpu::{run_webgpu, setup_webgpu};
use crate::halo2curves::utils::cast_u8_to_u16;
use wgpu::util::DeviceExt;


pub const WORKGROUP_SIZE: usize = 64;
pub const MAX_NUM_INVOCATIONS: usize = 1024;

pub async fn run_msm_inner(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8], device: &Device, queue: &Queue) -> Buffer {
    let msm_len = scalars_bytes.len() / 64;
    println!("msm_len: {:?}", msm_len);
    let num_invocations = (msm_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    println!("num_invocations: {:?}", num_invocations);
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
    let result_buffer_size = (MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let buckets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buckets Buffer"),
        contents: &vec![0u8; 32 * 256 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let windows = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Windows Buffer"),
        contents: &vec![0u8; 32 * MAX_NUM_INVOCATIONS * 3 * NUM_LIMBS * 4],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());
    let num_invocations_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Num Invocations Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&num_invocations_buffer, 0, &(num_invocations as u32).to_le_bytes());


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
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_buffer_size);
    };

    run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer,
            scalars_buffer,
            result_buffer.clone(),
            buckets,
            windows,
        ],
        vec![
            msm_len_buffer,
            num_invocations_buffer
        ],
        vec![
            ("run_bucket_accumulation_phase".to_string(), num_invocations as u32),
            ("run_bucket_reduction_phase".to_string(), num_invocations as u32),
            ("run_final_reduction_phase".to_string(), num_invocations as u32),
            ("aggregate".to_string(), 1),
        ], 
        compute_pipeline_fn,
        readback_buffer.clone(),
        copy_results_to_encoder,
    )
    .await
}

pub async fn run_msm(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let result = run_msm_inner(wgsl_source, points_bytes, scalars_bytes, &device, &queue).await;

    let buffer_slice = result.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();

    let output_u16 = cast_u8_to_u16(&data);
    drop(data);
    result.unmap();

    output_u16
}


pub async fn run_msm_browser(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let result = run_msm_inner(wgsl_source, points_bytes, scalars_bytes, &device, &queue).await;

    let buffer_slice = result.slice(..);
    // let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    let _ = map_buffer_async_browser(buffer_slice, wgpu::MapMode::Read).await;
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();

    let output_u16 = cast_u8_to_u16(&data);
    drop(data);
    result.unmap();

    output_u16 
}

pub async fn map_buffer_async_test(
    slice: BufferSlice<'_>,
    mode: MapMode,
) -> Result<(), BufferAsyncError> {
    let _buffer_future = slice.map_async(mode, |x| x.unwrap());
    Ok(())
}

pub fn map_buffer_async_browser(
    slice: BufferSlice<'_>,
    mode: MapMode,
) -> impl std::future::Future<Output = Result<(), BufferAsyncError>> {
    let (sender, receiver) = oneshot::channel();
    slice.map_async(mode, move |res| {
        let _ = sender.send(res);
    });
    async move {
        match receiver.await {
            Ok(result) => result,
            Err(_) => Err(BufferAsyncError {}),
        }
    }
}