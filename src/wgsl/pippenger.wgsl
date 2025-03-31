
// Bit length of the scalar
const B = 256u;
// Window chunk size in bits
const ChunkSize = 8u;
// Number of windows (B/C)
const NumWindows = 32u;
const BucketsPerWindow = 2u << ChunkSize; // 256
const TotalBuckets = BucketsPerWindow * NumWindows; // 256 * 32 = 8192
const PointsPerInvocation = 64u;

// 2^C
const TWO_POW_C: BigInt256 = BigInt256(
    array(256u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

@group(0) @binding(3)
var<storage, read_write> buckets: array<JacobianPoint, 256u * 32u>;
@group(0) @binding(4)
var<storage, read_write> msm_len: u32;

// n - number of points/scalars
// i - a point/scalar index, 1...n
// j - a window index, 0...W-1
// k - a bucket index, 0...2^C - 1
// s_i[j] - value of the j-th chunk of the i-th scalar
// B[j, k] - accumulator bucket

fn field_to_bytes(s: ScalarField) -> array<u32, 2 * N> {
    // Declare a local array of 32 elements
    var res: array<u32, 2 * N>;

    for (var i = 0u; i < 32u; i = i + 1u) {
       res[i] = 0u;
    }

    // For each 16-bit limb, split into two 8-bit bytes
    for (var i: u32 = 0u; i < N; i = i + 1u) {
        let limb: u32 = s.limbs[i];

        // Low 8 bits
        let low_byte:  u32 = limb & 0xFFu;

        // Next 8 bits
        let next_byte: u32 = (limb >> 8u) & 0xFFu;

        // Store the two bytes in consecutive array slots
        res[2u * i] = low_byte;
        res[2u * i + 1u] = next_byte;
    }

    return res;
}

// There will be NUM_INVOCATIONS invocations (workgroups) of this function, each with a different gidx
fn pippenger(gidx: u32) -> JacobianPoint {
    let base = gidx * PointsPerInvocation;
    for (var b = 0u; b < PointsPerInvocation; b = b + 1u) {
        buckets[base + b] = JACOBIAN_IDENTITY;
    }

    var windows: array<JacobianPoint, NumWindows>; 
    for (var j: u32 = 0u; j < NumWindows; j = j + 1u) {
        windows[j] = JACOBIAN_IDENTITY;
    }

    // Bucket accumulation
    for (var i = 0u; i < msm_len; i = i + 1u) {
        var scalar = scalars[base + i];
        var u8_scalar = field_to_bytes(scalar);

        var point = points[base + i];
        for (var j = 0u; j < NumWindows; j = j + 1u) {
            var s_j = u8_scalar[j];
            if (s_j != 0u) {
                let bucket_index = j * BucketsPerWindow + s_j;
                buckets[bucket_index] = jacobian_add(buckets[bucket_index], point);
            }
        }
    }

    // Bucket reduction
    for (var j=0u; j < NumWindows; j = j+1u) {
        var sum = JACOBIAN_IDENTITY;
        var sum_of_sums = JACOBIAN_IDENTITY;
        var k = BucketsPerWindow - 1u;
        loop {
            sum = jacobian_add(sum, buckets[j * BucketsPerWindow + k]);
            sum_of_sums = jacobian_add(sum_of_sums, sum);

            if (k == 1u) {
                break;
            }
            k = k - 1u;
        }

        windows[j] = sum_of_sums;
    }

    var result: JacobianPoint = JACOBIAN_IDENTITY;
    for (var j = NumWindows - 1; j >= 0; j = j - 1) {
        var tmp = jacobian_mul(result, TWO_POW_C);
        result = jacobian_add(
            windows[j],
            tmp
        );
    } 

    return result;
}