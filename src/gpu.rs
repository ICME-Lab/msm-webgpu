use wgpu::util::DeviceExt;
use wgpu::{BufferAsyncError, BufferSlice, MapMode};

const LIMB_WIDTH: usize = 16;
const BIGINT_SIZE: usize = 256;
const NUM_LIMBS: usize = BIGINT_SIZE / LIMB_WIDTH;
const WORKGROUP_SIZE: usize = 64;
const NUM_INVOCATIONS: usize = 1;
const MSM_SIZE: usize = WORKGROUP_SIZE * NUM_INVOCATIONS;

pub async fn run_msm_compute(
    wgsl_source: &str,
    points_bytes: &[u8],
    scalars_bytes: &[u8],
) -> Vec<u16> {
    let instance = wgpu::Instance::default();

    // Request an adapter (the GPU) from the browser
    let adapter = instance
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

    // println!("{:?}", adapter.get_info());

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
    let buffer1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buffer1"),
        size: (256 * NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let buffer2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buffer2"),
        size: (256 * NUM_INVOCATIONS * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let msm_len_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MSM Length Buffer"),
        contents: &((scalars_bytes.len() / 64) as u32).to_le_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // println!("msm_len: {:?}", scalars_bytes.len() / 32);

    // Create the Bind Group Layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MSM Bind Group Layout"),
        entries: &[
            // binding=0
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding=1
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding=2
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding=3
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding=4
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding=5
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create the pipeline layout and compute pipelines (`main` and `aggregate`)
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MSM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MSM Compute Pipeline (main)"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    //   let aggregate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    //     label: Some("MSM Compute Pipeline (aggregate)"),
    //     layout: Some(&pipeline_layout),
    //     module: &shader_module,
    //     entry_point: Some("aggregate"),
    //     compilation_options: wgpu::PipelineCompilationOptions::default(),
    //     cache: None,
    //   });

    // println!("points_bytes: {}, {:?}", points_bytes.len(), points_bytes);

    // Create the Bind Group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MSM Bind Group"),
        layout: &bind_group_layout,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mem.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffer1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffer2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: msm_len_buffer.as_entire_binding(),
            },
        ],
    });

    // Create a separate buffer for reading results back
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Encode commands: dispatch the main pipeline, then the aggregate pipeline, then copy out
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MSM Encoder"),
    });

    // a) Dispatch the main compute pass
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Main compute pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(NUM_INVOCATIONS as u32, 1, 1);
    }

    // b) Dispatch the aggregator pass
    //   {
    //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //       label: Some("Aggregate compute pass"),
    //       timestamp_writes: None,WebGPU
    //     });
    //     cpass.set_pipeline(&aggregate_pipeline);
    //     cpass.set_bind_group(0, &bind_group, &[]);
    //     cpass.dispatch_workgroups(1, 1, 1);
    //   }

    // c) Copy GPU result buffer into readback buffer
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_buffer_size);

    // Submit all commands
    queue.submit(Some(encoder.finish()));

    // Await the GPU queue to finish
    device.poll(wgpu::Maintain::Wait);
    let buffer_slice = readback_buffer.slice(..);

    // let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    map_buffer_async(buffer_slice, wgpu::MapMode::Read)
    .await
    .expect("map_async failed");


    device.poll(wgpu::Maintain::Wait);
    // Get the data
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

fn map_buffer_async(
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