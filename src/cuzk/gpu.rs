use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePipeline, ComputePipelineDescriptor, Device, Features, Instance, Limits,
    MapMode, MemoryHints, PipelineCompilationOptions, PipelineLayoutDescriptor, PowerPreference,
    Queue, ShaderModuleDescriptor, ShaderSource,
};

pub async fn get_adapter() -> Adapter {
    let instance = Instance::default();

    // Request an adapter (the GPU) from the browser
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapters found on the system!")
}

pub async fn get_device(adapter: &Adapter) -> (Device, Queue) {
    let required_limits = Limits {
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
    // let required_limits = adapter.limits();
    // let required_limits = wgpu::Limits { ..Default::default() };

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_limits: required_limits,
                required_features: Features::empty(),
                memory_hints: MemoryHints::default(), // Favor performance over memory usage
            },
            None,
        )
        .await
        .expect("Could not create adapter for device");

    (device, queue)
}

pub fn create_storage_buffer(label: Option<&str>, device: &Device, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: label,
        size: size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn create_and_write_storage_buffer(
    label: Option<&str>,
    device: &Device,
    data: &[u8],
) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: label,
        contents: data,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    })
}

pub fn create_and_write_uniform_buffer(
    label: Option<&str>,
    device: &Device,
    queue: &Queue,
    data: &[u8],
) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: label,
        size: data.len() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, data);

    buffer
}

pub fn read_from_gpu(
    device: &Device,
    queue: &Queue,
    mut encoder: CommandEncoder,
    storage_buffers: Vec<Buffer>,
    custom_size: u64,
) -> Vec<Vec<u8>> {
    let mut staging_buffers = Vec::new();

    for storage_buffer in storage_buffers {
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: storage_buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &storage_buffer,
            0,
            &staging_buffer,
            0,
            storage_buffer.size(),
        );
        staging_buffers.push(staging_buffer);
    }

    queue.submit(Some(encoder.finish()));

    println!("Staging buffers: {:?}", staging_buffers);

    println!("Before mapping");
    let mut data = Vec::new();
    for staging_buffer in staging_buffers {
        let staging_slice = staging_buffer.slice(..);
        let _buffer_future = staging_slice.map_async(MapMode::Read, |x| x.unwrap());
        device.poll(wgpu::Maintain::Wait);
        let result_data = staging_slice.get_mapped_range();
        data.push(result_data.to_vec());
    }

    println!("After mapping");

    data
}

pub fn create_bind_group_layout(
    label: Option<&str>,
    device: &Device,
    storage_buffers_read_only: Vec<&Buffer>,
    storage_buffers: Vec<&Buffer>,
    uniform_buffers: Vec<&Buffer>,
) -> BindGroupLayout {
    let storage_buffer_read_only_entries = (0..storage_buffers_read_only.len())
        .map(|i| default_storage_read_only_buffer_entry(i as u32))
        .collect::<Vec<_>>();
    let storage_buffer_entries = (0..storage_buffers.len())
        .map(|i| default_storage_buffer_entry((i + storage_buffers_read_only.len()) as u32))
        .collect::<Vec<_>>();

    let uniform_buffer_entries = (0..uniform_buffers.len())
        .map(|i| {
            default_uniform_buffer_entry(
                (i + storage_buffers.len() + storage_buffers_read_only.len()) as u32,
            )
        })
        .collect::<Vec<_>>();
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: label,
        entries: &vec![
            storage_buffer_read_only_entries,
            storage_buffer_entries,
            uniform_buffer_entries,
        ]
        .concat(),
    })
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

pub fn default_storage_read_only_buffer_entry(idx: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
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

pub fn create_bind_group(
    label: Option<&str>,
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    buffers: Vec<&Buffer>,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: label,
        layout: bind_group_layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

pub async fn create_compute_pipeline(
    label: Option<&str>,
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    code: &str,
    entry_point: &str,
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: label,
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: label,
        source: ShaderSource::Wgsl(code.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: label,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub async fn execute_pipeline(
    encoder: &mut CommandEncoder,
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    num_x_workgroups: u32,
    num_y_workgroups: u32,
    num_z_workgroups: u32,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    cpass.set_pipeline(&pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.dispatch_workgroups(num_x_workgroups, num_y_workgroups, num_z_workgroups);
}
