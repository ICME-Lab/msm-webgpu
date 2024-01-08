use std::borrow::Cow;
use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, Instance};

pub async fn device_setup_default(
    wgsl_source: &str,
) -> (
    wgpu::Instance,
    wgpu::Adapter,
    wgpu::Device,
    wgpu::Queue,
    wgpu::ComputePipeline,
    wgpu::CommandEncoder,
) {
    let instance = Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .expect("No gpu found");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Could not create adapter for device");

    println!("{:?}", adapter.get_info());
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
        }),
        entry_point: "main",
    });

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    (instance, adapter, device, queue, pipeline, encoder)
}

pub async fn run_cos_compute(wgsl_source: &str) {
    let (_, _, device, queue, pipeline, mut encoder) = device_setup_default(wgsl_source).await;

    let x: Vec<f32> = (0..1024).map(|v| v as f32).collect();

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"),
        contents: bytemuck::cast_slice(&x),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (x.len() * std::mem::size_of::<f32>()) as _,
        usage: wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buffer.as_entire_binding(),
            },
        ],
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(x.len() as u32, 1, 1);
    }

    queue.submit(Some(encoder.finish()));
    let buffer_slice = y_buffer.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();

    let res: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    y_buffer.unmap();

    let expected_result: Vec<f32> = (0..5).map(|x| f32::cos(x as _)).collect();

    println!("Result: {:?}", &res[0..5]);
    println!("expected result: {:?}", &expected_result);
}

pub async fn run_msm_compute(
    wgsl_source: &str,
    points_bytes: &[u8],
    scalars_bytes: &[u8],
    result_size: u64,
) -> Vec<u32> {
    let (_, _, device, queue, pipeline, mut encoder) = device_setup_default(wgsl_source).await;

    let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: points_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let scalars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scalars"),
        contents: scalars_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: result_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: result_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scalars_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());

    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();

    let res: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    res
}
