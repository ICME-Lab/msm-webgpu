@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: JacobianPoint;
@group(0) @binding(3)
var<storage, read_write> msm_len: u32;

@compute @workgroup_size(1)
fn test_point_msm(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    var running: JacobianPoint = JACOBIAN_IDENTITY;
    for (var i = 0u; i < msm_len; i = i + 1u) {
        let p = points[i];
        let s = scalars[i];
        let tmp = jacobian_mul(p, s);
        running = jacobian_add(running, tmp);
    }

    result = running;
}
