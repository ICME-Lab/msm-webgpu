use wgpu::{
    BindGroupLayoutEntry, Buffer, BufferAsyncError, BufferSlice, CommandEncoder, ComputePipeline, Device, MapMode, PipelineLayout, Queue
};
pub mod msm;
pub mod test;
pub const LIMB_WIDTH: usize = 16;
pub const BIGINT_SIZE: usize = 256;
pub const NUM_LIMBS: usize = BIGINT_SIZE / LIMB_WIDTH;

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
        max_storage_buffers_per_shader_stage: adapter.limits().max_storage_buffers_per_shader_stage,
        max_bind_groups: adapter.limits().max_bind_groups,
        max_bindings_per_bind_group: adapter.limits().max_bindings_per_bind_group,
        ..Default::default()
    };
    println!("Required limits: {:?}", required_limits);

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
    storage_buffers: Vec<Buffer>,
    uniform_buffers: Vec<Buffer>,
    pipeline_entry_points: Vec<(String, u32)>,
    compute_pipeline: impl Fn((String, PipelineLayout)) -> ComputePipeline,
    copy_results_to_encoder: impl Fn(&mut CommandEncoder) -> (),
) {

    let storage_buffer_entries =(0..storage_buffers.len())
        .map(|i| default_storage_buffer_entry(i as u32))
        .collect::<Vec<_>>();
    let uniform_buffer_entries =(0..uniform_buffers.len())
        .map(|i| default_uniform_buffer_entry((i + storage_buffers.len()) as u32))
        .collect::<Vec<_>>();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &vec![storage_buffer_entries, uniform_buffer_entries].concat(),
    });

    let buffers = vec![storage_buffers, uniform_buffers].concat();

    // Create the pipeline layout and compute pipelines (`main` and `aggregate`)
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MSM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipelines = pipeline_entry_points
        .iter()
        .map(|(entry_point, workgroups)| (compute_pipeline((entry_point.clone(), pipeline_layout.clone())), *workgroups))
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

    for (pipeline, workgroups) in compute_pipelines.iter() {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Main compute pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(*workgroups, 1, 1);
    }
    // c) Copy GPU result buffer into readback buffer
    // encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_buffer_size);
    copy_results_to_encoder(&mut encoder);

    // Submit all commands
    queue.submit(Some(encoder.finish()));

    // Await the GPU queue to finish
    // device.poll(wgpu::Maintain::Wait);
}



pub fn default_storage_buffer_entry(idx: u32) -> BindGroupLayoutEntry {
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
pub fn default_uniform_buffer_entry(idx: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}