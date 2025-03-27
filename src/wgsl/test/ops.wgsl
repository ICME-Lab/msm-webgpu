@group(0) @binding(0)
var<storage, read_write> a: ScalarField;
@group(0) @binding(1)
var<storage, read_write> b: ScalarField;
@group(0) @binding(2)
var<storage, read_write> result: ScalarField;

@compute @workgroup_size(1)
fn test_field_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result = field_mul(a, b);
}

@compute @workgroup_size(1)
fn test_field_add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result = field_add(a, b);
}

@compute @workgroup_size(1)
fn test_field_sub(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result = field_sub(a, b);
}