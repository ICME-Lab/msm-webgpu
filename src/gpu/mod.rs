use wgpu::util::DeviceExt;
use wgpu::{
    Adapter, BindGroupLayoutEntry, Buffer, BufferAsyncError, BufferSlice, BufferView, CommandEncoder, ComputePipeline, Device, Instance, MapMode, PipelineLayout, Queue
};

const LIMB_WIDTH: usize = 16;
const BIGINT_SIZE: usize = 256;
const NUM_LIMBS: usize = BIGINT_SIZE / LIMB_WIDTH;
const WORKGROUP_SIZE: usize = 64;
const NUM_INVOCATIONS: usize = 1;
const MSM_SIZE: usize = WORKGROUP_SIZE * NUM_INVOCATIONS;

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

pub async fn run_msm_compute(
    wgsl_source: &str,
    points_bytes: &[u8],
    scalars_bytes: &[u8],
) -> Vec<u16> {
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

    let readback_buffer = run_webgpu(
        &device,
        &queue,
        vec![points_buffer, scalars_buffer, result_buffer.clone(), mem, pippenger_pow_buffer, pippenger_sum_buffer, msm_len_buffer],
        vec!["main".to_string()], // TODO: add aggregate
        compute_pipeline_fn,
        readback_buffer.clone(),
        copy_results_to_encoder,
    ).await;

    let buffer_slice = readback_buffer.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();

    let output_u32: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    let output_u16 = output_u32
        .iter()
        .map(|&x| {
            if x > u16::MAX as u32 {
                panic!("Value {} is too large for u16", x);
            }
            x as u16
        })
        .collect::<Vec<_>>();
    drop(data);
    readback_buffer.unmap();

    output_u16
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
