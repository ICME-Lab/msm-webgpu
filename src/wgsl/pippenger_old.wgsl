const POINTS_PER_INVOCATION = 64u; // Each invocation (workgroup) handles 64 points.
const PARTITION_SIZE = 8u; // Each scalar is divided into 8-bit chunks (partitions).
const POW_PART = (1u << PARTITION_SIZE); // 2^8 = 256; Number of possible values per 8-bit partition.
const NUM_PARTITIONS = POINTS_PER_INVOCATION / PARTITION_SIZE; // 64 / 8 = 8; Partitions per workgroup.
const PS_SZ = POW_PART; // 2^8 = 256; Size of power set (aka sum combinations) = 256 per partition.
const BB_SIZE = 256; // Bucket size = 2^8 = 256.
const PS_SZ_NUM_INVOCATIONS = PS_SZ * 1u; // 256 * 4096 = 1048576
const BB_SIZE_NUM_INVOCATIONS = BB_SIZE * 1; // 256 * 4096 = 1048576
const NUM_WINDOWS = 32u; // 256/8


// For each workgroup, stores 256 precomputed combinations of points (i.e., all possible sums from 8 input points)
@group(0) @binding(4)
var<storage, read_write> powerset_sums: array<JacobianPoint, PS_SZ_NUM_INVOCATIONS>;
// Stores 256 buckets for accumulation during Pippenger.
@group(0) @binding(5)
var<storage, read_write> cur_sum: array<JacobianPoint, BB_SIZE_NUM_INVOCATIONS>;
@group(0) @binding(6)
var<storage, read_write> msm_len: u32;

fn get_window(scalar: ScalarField, j: u32) -> u32 {
    let byte_idx = j; // j in 0..31 (NUM_WINDOWS)
    let limb_idx = byte_idx / 4u;       // 4 bytes per u32
    let byte_offset = (byte_idx % 4u) * 8u;
    return (scalar.limbs[limb_idx] >> byte_offset) & 0xffu;
}

fn pippenger(gidx: u32) -> JacobianPoint {
    let ps_base = gidx * PS_SZ;            // Offset into powerset_sums
    let sum_base = i32(gidx) * BB_SIZE;         // Offset into cur_sum
    let point_base = gidx * POINTS_PER_INVOCATION;

    // Initialize buckets
    for(var bb = 0; bb < BB_SIZE; bb = bb + 1) {
        cur_sum[sum_base + bb] = JACOBIAN_IDENTITY;
    }

    for (var i = 0u; i < PS_SZ; i = i + 1u) {
        powerset_sums[ps_base + i] = JACOBIAN_IDENTITY;
    }

    // 8 partitions per workgroup, each with 8 points
    for (var part = 0u; part < NUM_PARTITIONS; part = part + 1u) {
        // Build all powerset combinations of 8 input points
        var idx = 0u;
        for (var j = 1u; j < POW_PART; j = j + 1u) {
            if ((i32(j) & -i32(j)) == i32(j)) {
                powerset_sums[ps_base + j] = points[point_base + part * PARTITION_SIZE + idx];
                idx = idx + 1;
            } else {
                let mask = j & u32(j - 1);
                let other_mask = j ^ mask;
                powerset_sums[ps_base + j] = jacobian_add(
                    powerset_sums[ps_base + mask],
                    powerset_sums[ps_base + other_mask]
                );
            }
        }

        // Accumulate into buckets
        for (var w = 0u; w < NUM_WINDOWS; w = w + 1u) {
        var powerset_idx = 0u;
        for (var j = 0u; j < PARTITION_SIZE; j = j + 1u) {
            let window = get_window(scalars[point_base + part * PARTITION_SIZE + j], w);
            if (window != 0u) {
                powerset_idx |= (1u << j);
            }
        }

            // Skip empty powerset index = 0 (optional, since it's identity)
            if (powerset_idx > 0u) {
                cur_sum[u32(sum_base) + powerset_idx] = jacobian_add(
                    cur_sum[u32(sum_base) + powerset_idx],
                    powerset_sums[ps_base + powerset_idx]
                );
            }
        }
    }

    // Final reduction: sum all buckets in reverse order
    var running_total: JacobianPoint = JACOBIAN_IDENTITY;
    for (var bb = i32(BB_SIZE) - 1; bb >= 0; bb = bb - 1) {
        running_total = jacobian_add(
            jacobian_double(running_total),
            cur_sum[sum_base + bb]
        );
    }

    return running_total;
}
