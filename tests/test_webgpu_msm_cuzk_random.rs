#[cfg(test)]
mod tests_wasm_pack_16 {
    use msm_webgpu::tests_wasm_pack::test_webgpu_msm_cuzk;
    use rand::Rng;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_cuzk_random() {
        // Random between 2^16 and 2^20
        let sample_size = rand::thread_rng().gen_range(1 << 16..1 << 20);
        test_webgpu_msm_cuzk(sample_size).await;
    }
}
