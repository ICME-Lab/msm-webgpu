struct MsmLen {
    val: u32,
}

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;

@group(0) @binding(2)
var<storage, read_write> buckets: array<JacobianPoint, TotalBuckets>;

@group(0) @binding(3)
var<uniform> msm_len: MsmLen;

@compute @workgroup_size(1)
fn test_bucket_accumulation(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    bucket_accumulation_phase(gidx);
}