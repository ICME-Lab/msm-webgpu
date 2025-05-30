self.importScripts("./pkg/msm_webgpu.js");

async function initWasm() {
    await wasm_bindgen("./pkg/msm_webgpu_bg.wasm");
}

initWasm().then(() => {
    self.onmessage = async (event) => {
        const data = event.data;

        function logOutput(message) {
            self.postMessage({ type: "log", message });
        }

        try {
            // Check the type of operation to perform
            if (data.type === "runCPU") {
                const { numberOfMSM } = data;

                console.log("Starting CPU MSM...");
                const result = await wasm_bindgen.run_cpu_msm_web(numberOfMSM, logOutput);
                self.postMessage({ type: "CPUResult", result });
            }
            else if (data.type === "runGPU") {
                const { numberOfMSM } = data;

                console.log("Starting GPU MSM...");
                const result = await wasm_bindgen.run_webgpu_msm_web(numberOfMSM, logOutput);
                console.log("GPU MSM execution completed.");
                self.postMessage({ type: "GPUResult", result });
            }
        } catch (err) {
            console.error("Execution error:", err);
            self.postMessage({ type: "error", error: err.toString() });
        }
    };
});