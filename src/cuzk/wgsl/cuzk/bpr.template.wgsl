{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> barrett_funcs }}
{{> bigint_funcs }}
{{> ec_funcs }}

/// Used as input buffers for the bucket sums from SMVP, but also repurposed to
/// store the m points.
@group(0) @binding(0)
var<storage, read_write> bucket_sum_x: array<BigInt>;
@group(0) @binding(1)
var<storage, read_write> bucket_sum_y: array<BigInt>;
@group(0) @binding(2)
var<storage, read_write> bucket_sum_z: array<BigInt>;

/// Output buffers to store the g points.
@group(0) @binding(3)
var<storage, read_write> g_points_x: array<BigInt>;
@group(0) @binding(4)
var<storage, read_write> g_points_y: array<BigInt>;
@group(0) @binding(5)
var<storage, read_write> g_points_z: array<BigInt>;

// Unfiform storage buffer.
@group(0) @binding(6)
var<uniform> params: vec3<u32>;


fn load_bucket_sum(idx: u32) -> Point {
    return Point(
        bucket_sum_x[idx],
        bucket_sum_y[idx],
        bucket_sum_z[idx]
    );
}

@compute
@workgroup_size({{ workgroup_size }})
fn stage_1(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    let thread_id = global_id.x; 
    let num_threads_per_subtask = {{ workgroup_size }}u;

    let subtask_idx = params[0]; 
    let num_columns = params[1]; 

    /// Number of subtasks per shader invocation (must be power of 2).
    let num_subtasks_per_bpr = params[2]; 

    /// Number of buckets per subtask.
    let num_buckets_per_subtask = num_columns / 2u; 
    let num_buckets_per_bpr = num_buckets_per_subtask * num_subtasks_per_bpr;

    /// Number of buckets to reduce per thread.
    let buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;

    let multiplier = subtask_idx + (thread_id / num_threads_per_subtask);
    let offset = num_buckets_per_subtask * multiplier;
    var idx = offset;

    if (thread_id % num_threads_per_subtask != 0u) {
        idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * 
              buckets_per_thread + offset;
    }

    var m = load_bucket_sum(idx);
    var g = m;
    for (var i = 0u; i < buckets_per_thread - 1u; i++) {
        let idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * 
                  buckets_per_thread - 1u - i;
        let bi = offset + idx;
        let b = load_bucket_sum(bi);
        m = point_add(m, b);
        g = point_add(g, m);
    }

    bucket_sum_x[idx] = m.x;
    bucket_sum_y[idx] = m.y;
    bucket_sum_z[idx] = m.z;

    let t = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    g_points_x[t] = g.x;
    g_points_y[t] = g.y;
    g_points_z[t] = g.z;

    // {{{ recompile }}}
}

@compute
@workgroup_size({{ workgroup_size }})
fn stage_2(@builtin(global_invocation_id) global_id: vec3<u32>) {    
    let thread_id = global_id.x; 
    let num_threads_per_subtask = {{ workgroup_size }}u;

    let subtask_idx = params[0]; 
    let num_columns = params[1]; 
    let num_subtasks_per_bpr = params[2]; 
    let num_buckets_per_subtask = num_columns / 2u;
    let num_buckets_per_bpr = num_buckets_per_subtask * num_subtasks_per_bpr;

    /// Number of buckets to reduce per thread.
    let buckets_per_thread = num_buckets_per_subtask / 
                             num_threads_per_subtask;

    let multiplier = subtask_idx + (thread_id / num_threads_per_subtask);
    let offset = num_buckets_per_subtask * multiplier;

    var idx = offset;
    if (thread_id % num_threads_per_subtask != 0u) {
        idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * 
              buckets_per_thread + offset;
    }

    var m = load_bucket_sum(idx);

    let t = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    var g = Point(
        g_points_x[t],
        g_points_y[t],
        g_points_z[t],
    );

    /// Perform scalar mul on m and add the result to g.
    let s = buckets_per_thread * (num_threads_per_subtask - (thread_id % num_threads_per_subtask) - 1u);
    g = point_add(g, double_and_add(m, s));

    g_points_x[t] = g.x;
    g_points_y[t] = g.y;
    g_points_z[t] = g.z;

    {{{ recompile }}}
}