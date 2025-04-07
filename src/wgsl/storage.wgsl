const WORKGROUP_SIZE = 64u;
const MAX_NUM_INVOCATIONS = 1280u;

struct MsmLen {
    val: u32,
}

@group(0) @binding(5)
var<uniform> msm_len: MsmLen;

struct NumInvocations {
    val: u32,
}

@group(0) @binding(6)
var<uniform> num_invocations: NumInvocations;

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: array<JacobianPoint, MAX_NUM_INVOCATIONS>;
@group(0) @binding(3)
var<storage, read_write> buckets: array<JacobianPoint, MAX_NUM_INVOCATIONS * TotalBuckets>;
@group(0) @binding(4)
var<storage, read_write> windows: array<JacobianPoint, MAX_NUM_INVOCATIONS * NumWindows>;


