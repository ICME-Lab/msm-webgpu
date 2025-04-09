// 3 -> 2
// storage -> workgroup
// second stage

// NUM_INVOCATIONS workgroups of 1 thread each
@compute @workgroup_size(1)
fn run_bucket_accumulation_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    if (gidx >= msm_len.val) {
        return;
    }

    bucket_accumulation_phase(gidx);
}

@compute @workgroup_size(1)
fn run_bucket_reduction_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;
    if (gidx >= msm_len.val) {
        return;
    }
    bucket_reduction_phase(gidx);
}

@compute @workgroup_size(1)
fn run_final_reduction_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;
    if (gidx >= msm_len.val) {
        return;
    }
    result[gidx] = final_reduction_phase(gidx);
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
    for (var j = 1u; j < split; j = j + 1u) {
        result[lidx] = jacobian_add(result[lidx], result[lidx + split * 256u]);
    }


    // Make sure all partial results are written
    storageBarrier();

    for (var offset: u32 = 256 / 2u; offset > 0u; offset = offset / 2u) {
        if (lidx < offset) {
            result[gidx] = jacobian_add(result[gidx], result[gidx + offset]);
        }
        storageBarrier();
    }
}
