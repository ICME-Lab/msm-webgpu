struct MsmLen {
    val: u32,
}

@group(0) @binding(0)
var<storage, read_write> result: array<u32, 2>;


@group(0) @binding(1)
var<uniform> msm_len: MsmLen;

struct NumInvocations {
    val: u32,
}

@group(0) @binding(2)
var<uniform> num_invocations: NumInvocations;


@compute @workgroup_size(1)
fn test_constants(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result[0] = msm_len.val;
    result[1] = num_invocations.val;
}