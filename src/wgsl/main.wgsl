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


    let split = num_invocations.val / 256u; // Each thread handles this many entries

    // Step 1: Each thread accumulates its vertical slice
    for (var j = 0u; j < split; j = j + 1u) {
        result[lidx] = jacobian_add(result[lidx], result[lidx + j * 256u]);
    }


    // Make sure all partial results are written
    storageBarrier();
    if (lidx == 0u) {
        var current_count = num_invocations.val;

        loop {
            if (current_count <= 1u) {
                break;
            }

            let half = current_count / 2u;

            for (var i = 0u; i < half; i = i + 1u) {
                result[i] = jacobian_add(result[i], result[i + half]);
            }

            current_count = half;
        }
    }
}
