@group(0) @binding(0)
var<storage, read_write> a: ScalarField;
@group(0) @binding(1)
var<storage, read_write> result: array<u32, 2 * N>;

@compute @workgroup_size(1)
fn test_field_to_bytes(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    result = to_bytes(a);
}


