#[cfg(test)]
mod tests_wasm_pack_16 {
    use msm_webgpu::tests_wasm_pack::test_webgpu_msm_cuzk_pallas;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_cuzk_16_pallas() {
        test_webgpu_msm_cuzk_pallas(1 << 16).await;
    }
}
