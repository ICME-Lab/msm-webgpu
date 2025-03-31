
// Bit length of the scalar
const B = 256u;
// Window chunk size in bits
const ChunkSize = 8u;
// Number of windows (B/C)
const NumWindows = 32u;
const BucketSizePerWindow = 2u << ChunkSize; // 256
const BucketSize = BucketSizePerWindow * NumWindows; // 256 * 32 = 8192

@group(0) @binding(4)
var<storage, read_write> powerset_sums: array<JacobianPoint, 256u * 1u>;
@group(0) @binding(5)
var<storage, read_write> cur_sum: array<JacobianPoint, 256 * 1>;
@group(0) @binding(6)
var<storage, read_write> msm_len: u32;

// n - number of points/scalars
// i - a point/scalar index, 1...n
// j - a window index, 0...W-1
// k - a bucket index, 0...2^C - 1
// s_i[j] - value of the j-th chunk of the i-th scalar
// B[j, k] - accumulator bucket

// There will be NUM_INVOCATIONS invocations (workgroups) of this function, each with a different gidx
fn pippenger(gidx: u32) -> JacobianPoint {
    let sum_base = gidx * BucketSizePerWindow;

    for (var b = 0u; b < BucketSizePerWindow; b = b + 1u) {
        cur_sum[sum_base + b] = JACOBIAN_IDENTITY;
    }

    // Accumulation phase: assign each point to a bucket based on scalar chunks
    for (var i = 0u; i < msm_len; i = i + 1u) {
        for (var j = 0u; j < NumWindows; j = j + 1u) {
            // Extract the j-th 8-bit chunk of scalar[i]
            let limb_idx = (j * ChunkSize) / 32u;
            let bit_offset = (j * ChunkSize) % 32u;

            let limb = scalars[i].limbs[limb_idx];
            let s_i_j = (limb >> bit_offset) & ((1u << ChunkSize) - 1u);            // Compute the bucket index for this window and chunk
            // Add the point to its corresponding bucket (cur_sum is global, sum_base scopes this workgroup)
            let bucket_index = sum_base + j * B + s_i_j;
            if (s_i_j != 0u) {
                cur_sum[bucket_index] = jacobian_add(cur_sum[bucket_index], points[i]);
            }
        }
    }

    // Step 3: Sum buckets using "sum of sums" per window
    var running_total: JacobianPoint = JACOBIAN_IDENTITY;

    // Final reduction: reverse-scan buckets and combine via double-and-add
    for (var j = 0u; j < NumWindows; j = j + 1u) {
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        for (var k: i32 = i32((1 << ChunkSize) - 1); k > 0; k = k - 1) {
            let idx = sum_base + j * B + u32(k);
            sum = jacobian_add(sum, cur_sum[idx]);
            sum_of_sums = jacobian_add(sum_of_sums, sum);
        }
        // sum_of_sums now holds W_j
        for (var d = 0u; d < ChunkSize; d = d + 1u) {
            running_total = jacobian_double(running_total);
        }

        running_total = jacobian_add(running_total, sum_of_sums);
    }

    return running_total;
}