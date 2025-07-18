#[cfg(test)]
mod tests_wasm_pack_17 {
    use msm_webgpu::tests_wasm_pack::test_webgpu_msm_cuzk;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_cuzk_17() {
        test_webgpu_msm_cuzk(1 << 17).await;
    }
}
