#[cfg(test)]
mod tests_wasm_pack_19 {
    use msm_webgpu::tests_wasm_pack::test_webgpu_msm_cuzk_bn256;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_webgpu_msm_cuzk_19() {
        test_webgpu_msm_cuzk_bn256(1 << 19).await;
    }
}
