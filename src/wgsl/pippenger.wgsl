
// n - number of points/scalars
// i - a point/scalar index, 1...n
// j - a window index, 0...W-1
// k - a bucket index, 0...2^C - 1
// s_i[j] - value of the j-th chunk of the i-th scalar
// B[j, k] - accumulator bucket

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

fn bucket_accumulation_phase(gidx: u32) {
    let base = gidx * PointsPerInvocation;
    for (var i = 0u; i < PointsPerInvocation; i = i + 1u) {
        // TODO: Revise this. Maybe pad with identity points
        if (msm_len.val > base + i) {
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
}


fn bucket_reduction_phase(gidx: u32) {
    for (var j: u32 = 0u; j < NumWindows; j = j + 1u) {    
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        for (var offset: u32 = 0u; offset < BucketsPerWindow - 1u; offset = offset + 1u) {
            let k = BucketsPerWindow - 1u - offset;
            let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + k;
            var bucket = buckets[bucket_index];
            sum = jacobian_add(bucket, sum);
            sum_of_sums = jacobian_add(sum, sum_of_sums);
        }
        windows[gidx * NumWindows + j] = sum_of_sums;
    }
}

fn final_reduction_phase(gidx: u32) -> JacobianPoint {
    var result: JacobianPoint = JACOBIAN_IDENTITY;

    var j = NumWindows - 1u;
    loop {
        result = jacobian_add(
            windows[gidx * NumWindows + j],
            jacobian_mul(result, TWO_POW_C)
        );
        if (j == 0u) {
            break;
        }
        j = j - 1u;
    }

    return result;
}


// There will be NUM_INVOCATIONS invocations (workgroups) of this function, each with a different gidx
