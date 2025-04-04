use wgpu::util::DeviceExt;

use crate::gpu::{setup_webgpu, run_webgpu};
use crate::gpu::*;
use crate::halo2curves::utils::cast_u8_to_u16;

// ------------------------------------------------------------
// Field operations
// ------------------------------------------------------------

pub async fn field_op(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8], op: &str) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Field Op Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let scalar_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scalar a Buffer"),
        contents: a_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let scalar_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scalar b Buffer"),
        contents: b_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Field Mul Compute Pipeline (main)"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0,  (NUM_LIMBS * 4) as wgpu::BufferAddress);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            scalar_a,
            scalar_b,
            result_buffer.clone(),
        ],
        vec![],
        vec![(op.to_string(), 1)], 
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

pub async fn field_mul(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u16> {
    field_op(wgsl_source, a_bytes, b_bytes, "test_field_mul").await
}

pub async fn field_add(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u16> {
    field_op(wgsl_source, a_bytes, b_bytes, "test_field_add").await
}

pub async fn field_sub(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u16> {
    field_op(wgsl_source, a_bytes, b_bytes, "test_field_sub").await
}


// ------------------------------------------------------------
// Point operations
// ------------------------------------------------------------


pub async fn point_op(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8], op: &str) -> Vec<u16> {
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Point Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let point_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Point a Buffer"),
        contents: a_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let point_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Point b Buffer"),
        contents: b_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Point Op Compute Pipeline (main)"),
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

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            point_a,
            point_b,
            result_buffer.clone(),
        ],
        vec![],
        vec![(op.to_string(), 1)], 
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

pub async fn point_add(wgsl_source: &str, a_bytes: &[u8], b_bytes: &[u8]) -> Vec<u16> {
    point_op(wgsl_source, a_bytes, b_bytes, "test_point_add").await
}

pub async fn point_double(wgsl_source: &str, a_bytes: &[u8]) -> Vec<u16> {
    point_op(wgsl_source, a_bytes, a_bytes, "test_point_double").await
}

pub async fn point_identity(wgsl_source: &str, a_bytes: &[u8]) -> Vec<u16> {
    point_op(wgsl_source, a_bytes, a_bytes, "test_point_identity").await
}


pub async fn point_msm(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let msm_len = scalars_bytes.len() / 64;
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Point Shader"),
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
    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Point Op Compute Pipeline (main)"),
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

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer.clone(),
            scalars_buffer.clone(),
            result_buffer.clone(),
        ],
        vec![
            msm_len_buffer.clone()
        ],
        vec![("test_point_msm".to_string(), 1)], 
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


// ------------------------------------------------------------


pub async fn field_to_bytes(wgsl_source: &str, a_bytes: &[u8]) -> Vec<u8> {
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Field Op Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
    });

    let scalar_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scalar a Buffer"),
        contents: a_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });


    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: (128) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: (128) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Field Mul Compute Pipeline (main)"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0,  (128) as wgpu::BufferAddress);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            scalar_a,
            result_buffer.clone(),
        ],
        vec![],
        vec![("test_field_to_bytes".to_string(), 1)], 
        compute_pipeline_fn,
        readback_buffer.clone(),
        copy_results_to_encoder,
    )
    .await;

    let buffer_slice = result.slice(..);
    let _buffer_future = buffer_slice.map_async(wgpu::MapMode::Read, |x| x.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let data = buffer_slice.get_mapped_range();

    let output = data.to_vec();
    drop(data);
    result.unmap();

    output
}


// ------------------------------------------------------------



pub async fn sum_of_sums_simple(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let msm_len = scalars_bytes.len() / 64;
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Point Shader"),
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
    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let windows = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Windows Buffer"),
        contents: &vec![0u8; 32 * 3 * NUM_LIMBS * 4],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Point Op Compute Pipeline (main)"),
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

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer.clone(),
            scalars_buffer.clone(),
            result_buffer.clone(),
            windows
        ],
        vec![
            msm_len_buffer.clone()
        ],
        vec![("test_sum_of_sums_simple".to_string(), 1)], 
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


pub async fn sum_of_sums(wgsl_source: &str, points_bytes: &[u8], scalars_bytes: &[u8]) -> Vec<u16> {
    let msm_len = scalars_bytes.len() / 64;
    let (device, queue) = setup_webgpu().await;
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Point Shader"),
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
    let msm_len_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MSM Length Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&msm_len_buffer, 0, &(msm_len as u32).to_le_bytes());
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });        

    let windows = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Windows Buffer"),
        contents: &vec![0u8; 32 * 3 * NUM_LIMBS * 4],
        usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: 32 * 3 * (NUM_LIMBS * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });


    let compute_pipeline_fn = |(entry_point, pipeline_layout): (String, PipelineLayout)| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Point Op Compute Pipeline (main)"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let copy_results_to_encoder = |encoder: &mut CommandEncoder| {
        encoder.copy_buffer_to_buffer(&windows, 0, &readback_buffer, 0, (32 * 3 * NUM_LIMBS * 4) as wgpu::BufferAddress);
    };

    let result = run_webgpu(
        &device,
        &queue,
        vec![
            points_buffer.clone(),
            scalars_buffer.clone(),
            result_buffer.clone(),
            windows.clone()
        ],
        vec![
            msm_len_buffer.clone()
        ],
        vec![("test_sum_of_sums".to_string(), 1)], 
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