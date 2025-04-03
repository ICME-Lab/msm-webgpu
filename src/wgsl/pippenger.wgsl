
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
    // for (var b = 0u; b < TotalBuckets; b = b + 1u) {
    //     buckets[gidx * TotalBuckets + b] = JACOBIAN_IDENTITY;
    // }
    let base = gidx * PointsPerInvocation;
    for (var i = 0u; i < PointsPerInvocation; i = i + 1u) {
        // TODO: Revise this. Maybe pad with identity points
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


fn bucket_reduction_phase(gidx: u32) {
    for (var j: u32 = 0u; j < NumWindows; j = j + 1u) {    
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        // var k = BucketsPerWindow - 1u;
        // for (var offset: u32 = 0u; offset < BucketsPerWindow - 1u; offset = offset + 1u) {
        //     let k = BucketsPerWindow - 1u - offset;
        //     let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + k;
        //     var tmp_sum = jacobian_add(sum, buckets[bucket_index]);
        //     var tmp_sum_of_sums = jacobian_add(sum_of_sums, tmp_sum);
        //     sum = tmp_sum;
        //     sum_of_sums = tmp_sum_of_sums;
        // }
        for (var k: i32 = i32(BucketsPerWindow - 1u); k >= 1; k = k - 1) {
            let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + u32(k);
            var bucket = buckets[bucket_index];
            sum = jacobian_add(bucket, sum);
            sum_of_sums = jacobian_add(sum, sum_of_sums);
        }
        // loop {
        //     let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + k;
        //     let b = buckets[bucket_index];

        //     sum = jacobian_add(sum, buckets[bucket_index]);
        //     workgroupBarrier();
        //     // sum_of_sums = jacobian_add(sum_of_sums, sum);
        //     sum_of_sums = sum; //jacobian_add(sum_of_sums, sum);
        //     workgroupBarrier();

        //     if (k == 1u) {
        //         break;
        //     }
        //     k = k - 1u;
        // }

        // var sum_of_sums = JACOBIAN_IDENTITY;
        // for (var k: u32 = 1u; k < BucketsPerWindow; k = k + 1u) {
        //     let bucket_index = gidx * TotalBuckets + j * BucketsPerWindow + k;
        //     var bucket = buckets[bucket_index];
        //     var scalar = BigInt256(
        //         array(k, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
        //     );
        //     var tmp = small_jacobian_mul(bucket, scalar);
        //     sum_of_sums = jacobian_add(
        //         sum_of_sums, 
        //         tmp
        //     );
        // }

        windows[gidx * NumWindows + j] = sum_of_sums;
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
    // var windows: array<JacobianPoint, NumWindows>; 


    // Bucket accumulation
    bucket_accumulation_phase(gidx);

    // Bucket reduction
    bucket_reduction_phase(gidx);

    var result: JacobianPoint = JACOBIAN_IDENTITY;

    var j = NumWindows - 1u;
    loop {
        var tmp = jacobian_mul(result, TWO_POW_C);
        result = jacobian_add(
            windows[gidx * NumWindows + j],
            tmp
        );
        if (j == 0u) {
            break;
        }
        j = j - 1u;
    }

    return result;
}