
// Bit length of the scalar
const B = 256u;
// Window chunk size in bits
const ChunkSize = 8u;
// Number of windows (B/C)
const NumWindows = 32u;
const BucketsPerWindow = 1u << ChunkSize; // 256
const TotalBuckets = BucketsPerWindow * NumWindows; // 256 * 32 = 8192
const PointsPerInvocation = 64u;

// 2^C
const TWO_POW_C: BigInt256 = BigInt256(
    array(256u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

@group(0) @binding(3)
var<storage, read_write> buckets: array<JacobianPoint, NUM_INVOCATIONS * PointsPerInvocation>;

struct MsmLen {
    val: u32,
}

@group(0) @binding(4)
var<uniform> msm_len: MsmLen;

struct NumInvocations {
    val: u32,
}

@group(0) @binding(5)
var<uniform> num_invocations: NumInvocations;

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

// n - number of points/scalars
// i - a point/scalar index, 1...n
// j - a window index, 0...W-1
// k - a bucket index, 0...2^C - 1
// s_i[j] - value of the j-th chunk of the i-th scalar
// B[j, k] - accumulator bucket

// There will be NUM_INVOCATIONS invocations (workgroups) of this function, each with a different gidx
fn pippenger(gidx: u32) -> JacobianPoint {
    var windows: array<JacobianPoint, NumWindows>; 
    for (var j: u32 = 0u; j < NumWindows; j = j + 1u) {
        windows[j] = JACOBIAN_IDENTITY;
    }

    // Bucket accumulation
    bucket_accumulation_phase(gidx);

    // Bucket reduction
    for (var j=0u; j < NumWindows; j = j+1u) {
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        var k = BucketsPerWindow - 1u;
        loop {
            let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + k;
            sum = jacobian_add(sum, buckets[bucket_index]);
            sum_of_sums = jacobian_add(sum_of_sums, sum);

            if (k == 1u) {
                break;
            }
            k = k - 1u;
        }

        windows[j] = sum_of_sums;
    }

    var result: JacobianPoint = JACOBIAN_IDENTITY;

    var j = NumWindows - 1u;
    loop {
        var tmp = jacobian_mul(result, TWO_POW_C);
        result = jacobian_add(
            windows[j],
            tmp
        );
        if (j == 0u) {
            break;
        }
        j = j - 1u;
    }

    return result;
}