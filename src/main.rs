use msm_webgpu::gpu;
fn main() {
    pollster::block_on(gpu::run_compute());
}
