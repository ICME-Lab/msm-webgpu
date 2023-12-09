use std::borrow::Cow;
use wgpu::util::DeviceExt;
use wgpu::Instance;

pub async fn device_setup_default() -> (
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

    let shader = "
      struct Array {
          data: array<f32>,
      }; 

      @group(0) 
      @binding(0)
      var<storage, read> x: Array;

      @group(0) 
      @binding(1)
      var<storage, read_write> y: Array;
          
      @compute
      @workgroup_size(1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let gidx = global_id.x;
          y.data[gidx] = cos(x.data[gidx]);
      }
    ";

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        }),
        entry_point: "main",
    });

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    (instance, adapter, device, queue, pipeline, encoder)
    // assert_eq!(&res[0..5], &expected_result);
}

pub async fn run_compute() {
    let (_, _, device, queue, pipeline, mut encoder) = device_setup_default().await;

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
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();

    let res: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    y_buffer.unmap();

    let expected_result: Vec<f32> = (0..5).map(|x| f32::cos(x as _)).collect();

    println!("Result: {:?}", &res[0..5]);
    println!("expected result: {:?}", &expected_result);
}
