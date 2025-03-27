use wgpu::{
    Adapter, BindGroupLayoutEntry, Buffer, BufferAsyncError, BufferSlice, BufferView, CommandEncoder, ComputePipeline, Device, Instance, MapMode, PipelineLayout, Queue
};
pub mod msm;
pub mod ops;

pub const LIMB_WIDTH: usize = 16;
pub const BIGINT_SIZE: usize = 256;
pub const NUM_LIMBS: usize = BIGINT_SIZE / LIMB_WIDTH;
pub const WORKGROUP_SIZE: usize = 64;
pub const NUM_INVOCATIONS: usize = 1;
pub const MSM_SIZE: usize = WORKGROUP_SIZE * NUM_INVOCATIONS;

pub async fn setup_webgpu() -> (Device, Queue) {
    let instance = wgpu::Instance::default();

    // Request an adapter (the GPU) from the browser
    let adapter: wgpu::Adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

    // Request the device and queue from the adapter
    let required_limits = wgpu::Limits {
        max_buffer_size: adapter.limits().max_buffer_size,
        max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
        max_compute_workgroup_storage_size: adapter.limits().max_compute_workgroup_storage_size,
        max_compute_workgroup_size_x: 1024,
        max_compute_invocations_per_workgroup: 1024,
        max_compute_workgroups_per_dimension: adapter.limits().max_compute_workgroups_per_dimension,
        ..Default::default()
    };

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_limits: required_limits,
                required_features: wgpu::Features::empty(),
                memory_hints: wgpu::MemoryHints::default(), // Favor performance over memory usage
            },
            None,
        )
        .await
        .expect("Could not create adapter for device");

    (device, queue)
}

pub async fn run_webgpu(
    device: &Device,
    queue: &Queue,
    buffers: Vec<Buffer>,
    pipeline_entry_points: Vec<String>,
    compute_pipeline: impl Fn((String, PipelineLayout)) -> ComputePipeline,
    readback_buffer: Buffer,
    copy_results_to_encoder: impl Fn(&mut CommandEncoder) -> (),
) -> Buffer {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &(0..buffers.len())
            .map(|i| default_bind_group_layout_entry(i as u32))
            .collect::<Vec<_>>(),
    });

    // Create the pipeline layout and compute pipelines (`main` and `aggregate`)
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MSM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipelines = pipeline_entry_points
        .iter()
        .map(|entry_point| compute_pipeline((entry_point.clone(), pipeline_layout.clone())))
        .collect::<Vec<_>>();

    // Create the Bind Group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MSM Bind Group"),
        layout: &bind_group_layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MSM Encoder"),
    });

    for (i, pipeline) in compute_pipelines.iter().enumerate() {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Main compute pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(NUM_INVOCATIONS as u32, 1, 1);
    }
    // c) Copy GPU result buffer into readback buffer
    // encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_buffer_size);
    copy_results_to_encoder(&mut encoder);

    // Submit all commands
    queue.submit(Some(encoder.finish()));

    // Await the GPU queue to finish
    device.poll(wgpu::Maintain::Wait);


    readback_buffer
}

pub fn map_buffer_async(
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

pub fn default_bind_group_layout_entry(idx: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
