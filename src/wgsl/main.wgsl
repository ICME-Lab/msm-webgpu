// 3 -> 2
// storage -> workgroup
// second stage
@group(0) @binding(6)
var<storage, read_write> msm_len: u32;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    var running: JacobianPoint = JACOBIAN_IDENTITY;
    for (var i = 0u; i < msm_len; i = i + 1u) {
        let p = points[i];
        let s = scalars[i];
        let tmp = jacobian_mul(p, s);
        running = jacobian_add(running, tmp);
    }
    result[gidx] = running;
}

// @compute @workgroup_size(256)
// fn aggregate(
//     @builtin(global_invocation_id) global_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>
// ) {
//     let gidx = global_id.x;
//     let lidx = local_id.x;

//     const split = NUM_INVOCATIONS / 256u;

//     for (var j = 1u; j < split; j = j + 1u) {
//         result[lidx] = jacobian_add(result[lidx], result[lidx + split * 256]);
//     }

//     storageBarrier();

//     for (var offset: u32 = 256u / 2u; offset > 0u; offset = offset / 2u) {
//         if (lidx < offset) {
//             result[gidx] = jacobian_add(result[gidx], result[gidx + offset]);
//         }
//         storageBarrier();
//     }
// }
