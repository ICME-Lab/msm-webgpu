@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: JacobianPoint;
struct MsmLen {
    val: u32,
}
@group(0) @binding(3)
var<storage, read_write> windows: array<JacobianPoint, 32>;

@group(0) @binding(4)
var<uniform> msm_len: MsmLen;


@compute @workgroup_size(1)
fn test_sum_of_sums_simple(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    var sum = JACOBIAN_IDENTITY;
    var sum_of_sums = JACOBIAN_IDENTITY;
            
    for (var k: i32 = i32(msm_len.val - 1u); k >= 0; k = k - 1) {
        var bucket = points[k];
        sum = jacobian_add(bucket, sum);
        sum_of_sums = jacobian_add(sum, sum_of_sums);
    }
    result = sum_of_sums;
}

@compute @workgroup_size(1)
fn test_sum_of_sums(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    for (var j= 0u; j<= 32u; j = j + 1) {
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        for (var k: i32 = i32(msm_len.val - 1u); k >= 0; k = k - 1) {
            var bucket = points[k];
            sum = jacobian_add(bucket, sum);
            sum_of_sums = jacobian_add(sum, sum_of_sums);
        }
        windows[j] = sum_of_sums;
    }
}
