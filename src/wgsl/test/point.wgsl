@group(0) @binding(0)
var<storage, read_write> a: JacobianPoint;
@group(0) @binding(1)
var<storage, read_write> b: JacobianPoint;
@group(0) @binding(2)
var<storage, read_write> result: JacobianPoint;

@compute @workgroup_size(1)
fn test_point_add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result = jacobian_add(a, b);
}

@compute @workgroup_size(1)
fn test_point_double(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result = jacobian_double(a);
}

@compute @workgroup_size(1)
fn test_point_identity(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    var identity: JacobianPoint = JACOBIAN_IDENTITY;

    result = jacobian_add(a, identity);
}