const WORKGROUP_SIZE = 64u;
const NUM_INVOCATIONS = 4096u;

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: array<JacobianPoint, NUM_INVOCATIONS>;
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
