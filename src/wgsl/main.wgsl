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


// Assuming a workgroup size of 256 (as in your original code)
@compute @workgroup_size(1)
fn aggregate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lidx = local_id.x;
    var n = num_invocations.val;
   
    if (lidx == 0) {
        while (n > 1) {
            let l = (n + 1) / 2;
            let pairs = n / 2;
            for (var i = 0u; i < pairs; i = i + 1u) {
                result[i] = jacobian_add(
                    result[2 * i],
                    result[2 * i + 1]
                );
            }
            if (n % 2 == 1) {
                result[pairs] = result[n - 1];
            }
            n = (n + 1) / 2;
        }
    }
}
