use msm_webgpu::{gpu, utils::concat_files};
fn main() {
    let shader_code = concat_files(
        vec!["src/wgsl/test_cos.wgsl"]
    );
    pollster::block_on(gpu::run_cos_compute(&shader_code));
}
