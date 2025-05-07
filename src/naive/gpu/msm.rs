use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use crate::naive::gpu::*;
use crate::naive::gpu::{run_webgpu, setup_webgpu};
use crate::naive::halo2curves::utils::cast_u8_to_u16;
use wgpu::util::DeviceExt;
use gloo_timers::future::sleep;


pub const WORKGROUP_SIZE: usize = 64;
pub const MAX_NUM_INVOCATIONS: usize = 1300;

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
        size: (3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
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
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, (3 * NUM_LIMBS * 4) as wgpu::BufferAddress);
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
        copy_results_to_encoder,
    )
    .await;

    readback_buffer
}

pub async fn run_msm(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let now = Instant::now();
    let result = run_msm_inner(wgsl_source, points_bytes, scalars_bytes, &device, &queue).await;
    println!("MSM time: {:?}", now.elapsed());

    let now = Instant::now();
    let buffer_slice = result.slice(0..(3 * NUM_LIMBS * 4) as u64);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();
    println!("Mapping time: {:?}", now.elapsed());

    let now = Instant::now();
    let output_u16 = cast_u8_to_u16(&data);
    println!("Casting time: {:?}", now.elapsed());
    drop(data);
    result.unmap();

    output_u16
}
use web_sys::console;
use wasm_bindgen::prelude::*;
use wgpu::BufferView;


#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

pub async fn run_msm_browser(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;

    let readback_buffer = run_msm_inner(wgsl_source, points_bytes, scalars_bytes, &device, &queue).await;

    let buffer_slice = readback_buffer.slice(0..(3 * NUM_LIMBS * 4) as u64);
    let _ = map_buffer_async_browser(buffer_slice, wgpu::MapMode::Read).await;
    // let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();

    let output_u16 = cast_u8_to_u16(&data);
    drop(data);
    readback_buffer.unmap();

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

async fn wait_for_mapping(buffer_slice: wgpu::BufferSlice<'_>) -> Result<BufferView<'_>, wgpu::BufferAsyncError> {
    // Start the async mapping
    buffer_slice.map_async(wgpu::MapMode::Read, |res| {
        // You may log or process the result here if needed.
        // Do not unwrap here; propagate the error if desired.
        if let Err(e) = res {
            web_sys::console::log_1(&format!("map_async error: {:?}", e).into());
        } else {
            res.unwrap()
        }
    });
    
    // Poll until the mapping is complete.
    // device.poll(wgpu::Maintain::Wait) can be called before awaiting.
    // But since that call blocks until all GPU work is done, in a browser you typically await the mapping future.
    // Here, instead of sleeping, try awaiting the mapping future:
    let mapping_result = async {
        loop {
            // Give control back to the browser so that the mapping callback can run.
            sleep(Duration::from_millis(10)).await;
            if let Ok(v) = safe_get_mapped_range(&buffer_slice) {
                return v;
            }
        }
    }.await;
    Ok(mapping_result)
}

fn safe_get_mapped_range<'a>(slice: &BufferSlice<'a>) -> Result<BufferView<'a>, Box<dyn Any + Send>> {
    catch_unwind(AssertUnwindSafe(|| {
        // This will panic if the slice is not mapped properly.
        slice.get_mapped_range()
    }))
}