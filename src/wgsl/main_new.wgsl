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


    let split = NUM_INVOCATIONS / 256u; // Each thread handles this many entries
    var sum = JACOBIAN_IDENTITY;

    // Step 1: Each thread accumulates its vertical slice
    // for (var j = 0u; j < split; j = j + 1u) {
    //     sum = jacobian_add(sum, result[lidx + j * 256u]);
    // }

    // // Step 2: Write partial sum into result[lidx]
    // result[lidx] = sum;

    // // Make sure all partial results are written
    // storageBarrier();

    // // Step 3: Parallel reduction to compute final result in result[0]
    // var offset = 256u / 2u;
    // while (offset > 0u) {
    //     if (lidx < offset) {
    //         result[lidx] = jacobian_add(result[lidx], result[lidx + offset]);
    //     }
    //     storageBarrier();
    //     offset = offset / 2u;
    // }
}
