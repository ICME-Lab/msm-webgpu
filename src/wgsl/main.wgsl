// 3 -> 2
// storage -> workgroup
// second stage

// NUM_INVOCATIONS workgroups of 1 thread each
@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    result[gidx] = pippenger(gidx);
}

// Only one workgroup of 256 threads
@compute @workgroup_size(256)
fn aggregate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    const split = NUM_INVOCATIONS / 256u;

    for (var j = 1u; j < split; j = j + 1u) {
        result[lidx] = jacobian_add(result[lidx], result[lidx + j * 256]);
    }

    // ensure that all threads have completed their memory operations before proceeding
    storageBarrier();

    for (var offset: u32 = 256u / 2u; offset > 0u; offset = offset / 2u) {
    if (lidx < offset) {
            result[lidx] = jacobian_add(result[lidx], result[lidx + offset]);
        }
        storageBarrier();
    }
}
