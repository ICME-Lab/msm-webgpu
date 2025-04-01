// Bit length of the scalar
const B = 256u;
// Window chunk size in bits
const ChunkSize = 8u;
// Number of windows (B/C)
const NumWindows = 32u;
const BucketsPerWindow = 1u << ChunkSize; // 256
const TotalBuckets = BucketsPerWindow * NumWindows; // 256 * 32 = 8192
const PointsPerInvocation = 64u;
const NumInvocations = 1u;

struct MsmLen {
    val: u32,
}

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;

@group(0) @binding(2)
var<storage, read_write> buckets: array<JacobianPoint, NumInvocations * PointsPerInvocation>;

@group(0) @binding(3)
var<uniform> msm_len: MsmLen;

fn bucket_accumulation_phase(gidx: u32) {
    for (var b = 0u; b < TotalBuckets; b = b + 1u) {
        buckets[gidx * TotalBuckets + b] = JACOBIAN_IDENTITY;
    }

    let base = gidx * PointsPerInvocation;
    for (var i = 0u; i < PointsPerInvocation; i = i + 1u) {
        if (msm_len.val < base + i) {
            break;
        }
        var scalar = scalars[base + i];
        var u8_scalar = field_to_bytes(scalar);
    
        var point = points[base + i];
        for (var j = 0u; j < NumWindows; j = j + 1u) {
            var s_j = u8_scalar[j];
            if (s_j != 0u) {
                let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + s_j;
                buckets[bucket_index] = jacobian_add(buckets[bucket_index], point);
            }
        }
    }
}

@compute @workgroup_size(1)
fn test_bucket_accumulation(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    bucket_accumulation_phase(gidx);
}