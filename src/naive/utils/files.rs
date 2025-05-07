pub fn load_shader_code_pallas() -> String {
    let mut all = String::new();
    all.push_str(include_str!("../wgsl/bigint.wgsl"));
    all.push_str(include_str!("../wgsl/pallas/field.wgsl"));
    all.push_str(include_str!("../wgsl/pallas/curve.wgsl"));
    all.push_str(include_str!("../wgsl/storage.wgsl"));
    all.push_str(include_str!("../wgsl/pippenger.wgsl"));
    all.push_str(include_str!("../wgsl/main.wgsl"));
    all
}

pub fn load_shader_code_bn254() -> String {
    let mut all = String::new();
    all.push_str(include_str!("../wgsl/bigint.wgsl"));
    all.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    all.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
    all.push_str(include_str!("../wgsl/storage.wgsl"));
    all.push_str(include_str!("../wgsl/pippenger.wgsl"));
    all.push_str(include_str!("../wgsl/main.wgsl"));
    all
}

pub fn load_shader_code(modulus: &str) -> String {
    if modulus == "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47" {
        load_shader_code_bn254()
    } else if modulus == "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001" {
        load_shader_code_pallas()
    } else {
        panic!("WebGPU not implemented for this curve: {}", modulus)
    }
}
// ------------------------------------------------------------
// BN254
// ------------------------------------------------------------

pub fn load_bn254_field_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/field.wgsl"));
    shader_code
}

pub fn load_bn254_point_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/point.wgsl"));
    shader_code
}

pub fn load_bn254_point_msm_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/msm.wgsl"));
    shader_code
}

pub fn load_bn254_sum_of_sums_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/sum_of_sums.wgsl"));
    shader_code
}

pub fn load_bn254_field_to_bytes_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/field_to_bytes.wgsl"));
    shader_code
}

pub fn load_bn254_constants_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/constants.wgsl"));
    shader_code
}

pub fn load_bn254_pippenger_phases_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/bn254/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pippenger.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/pippenger_phases.wgsl"));
    shader_code
}

// ------------------------------------------------------------
// Pallas
// ------------------------------------------------------------

pub fn load_pallas_field_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/field.wgsl"));
    shader_code
}

pub fn load_pallas_point_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pallas/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/point.wgsl"));
    shader_code
}

pub fn load_pallas_point_msm_shader_code() -> String {
    let mut shader_code = String::new();
    shader_code.push_str(include_str!("../wgsl/bigint.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pallas/field.wgsl"));
    shader_code.push_str(include_str!("../wgsl/pallas/curve.wgsl"));
    shader_code.push_str(include_str!("../wgsl/test/msm.wgsl"));
    shader_code
}
