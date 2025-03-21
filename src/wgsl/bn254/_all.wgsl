            // const shaders_in_order = [
            //     'bigint.wgsl',
            //     'field.wgsl',
            //     'curve.wgsl',
            //     'storage.wgsl',
            //     'pippenger.wgsl',
            //     // 'pippenger_fake.wgsl',
            //     'main.wgsl',
            // ];

// ---------

// bigint.wgsl

const W = 16u;
const W_mask = (1 << W) - 1u;
const L = 256u;
const N = 16u;
const NN = N * 2u;
const N1 = N + 1u;

// No overflow
struct BigInt256 {
    limbs: array<u32,N>
}

struct BigInt512 {
    limbs: array<u32,NN>
}

struct BigInt272 {
    limbs: array<u32,N1>
}

// Careful, a and res may point to the same thing.
fn add(a: BigInt256, b: BigInt256, res: ptr<function, BigInt256>) -> u32 {
    var carry: u32 = 0;
    for (var i: u32 = 0; i < N; i = i + 1u) {
        let c = a.limbs[i] + b.limbs[i] + carry;
        (*res).limbs[i] = c & W_mask;
        carry = c >> W;
    }
    return carry;
}
 
// assumes a >= b
fn sub(a: BigInt256, b: BigInt256, res: ptr<function, BigInt256>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < N; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

// repeated code pls fix
fn add_512(a: BigInt512, b: BigInt512, res: ptr<function, BigInt512>) -> u32 {
    var carry: u32 = 0;
    for (var i: u32 = 0; i < (2*N); i = i + 1u) {
        let c = a.limbs[i] + b.limbs[i] + carry;
        (*res).limbs[i] = c & W_mask;
        carry = c >> W;
    }
    return carry;
}
 
// assumes a >= b
fn sub_512(a: BigInt512, b: BigInt512, res: ptr<function, BigInt512>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < (2*N); i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

// assumes a >= b
fn sub_272(a: BigInt272, b: BigInt272, res: ptr<function, BigInt272>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < N + 1; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn mul(a: BigInt256, b: BigInt256) -> BigInt512 {
    var res: BigInt512;
    for (var i = 0u; i < N; i = i + 1u) {
        for (var j = 0u; j < N; j = j + 1u) {
            let c = a.limbs[i] * b.limbs[j];
            res.limbs[i+j] += c & W_mask;
            res.limbs[i+j+1] += c >> W;
        }   
    }
    // start from 0 and carry the extra over to the next index
    for (var i = 0u; i < 2*N - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> W;
        res.limbs[i] = res.limbs[i] & W_mask;
    }
    return res;
}

fn sqr(a: BigInt256) -> BigInt512 {
    var res: BigInt512;
    for (var i = 0u;i < N; i = i + 1u) {
        let sc = a.limbs[i] * a.limbs[i];
        res.limbs[(i << 1)] += sc & W_mask;
        res.limbs[(i << 1)+1] += sc >> W;

        for (var j = i + 1;j < N;j = j + 1u) {
            let c = a.limbs[i] * a.limbs[j];
            res.limbs[i+j] += (c & W_mask) << 1;
            res.limbs[i+j+1] += (c >> W) << 1;
        }
    }

    for (var i = 0u; i < 2*N - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> W;
        res.limbs[i] = res.limbs[i] & W_mask;
    }
    return res;
}

// field.wgsl

alias BaseField = BigInt256;
alias ScalarField = BigInt256;

const BASE_MODULUS: BigInt256 = BigInt256(
    array(64839u, 55420u, 35862u, 15392u, 51853u, 26737u, 27281u, 38785u, 22621u, 33153u, 17846u, 47184u, 41001u, 57649u, 20082u, 12388u)
);

const BASE_MODULUS_MEDIUM_WIDE: BigInt272 = BigInt272(
    array(64839u, 55420u, 35862u, 15392u, 51853u, 26737u, 27281u, 38785u, 22621u, 33153u, 17846u, 47184u, 41001u, 57649u, 20082u, 12388u, 0u)
);

const BASE_MODULUS_WIDE: BigInt512 = BigInt512(
    array(64839u, 55420u, 35862u, 15392u, 51853u, 26737u, 27281u, 38785u, 22621u, 33153u, 17846u, 47184u, 41001u, 57649u, 20082u, 12388u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

const BASE_M = BigInt256(
    array(2788u, 40460u, 53156u, 3965u, 54731u, 24120u, 21946u, 41466u, 40585u, 63994u, 59685u, 7870u, 32601u, 31545u, 50740u, 48750u)
);

const ZERO: BigInt256 = BigInt256(
    array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

const ONE: BigInt256 = BigInt256(
    array(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

fn get_higher_with_slack(a: BigInt512) -> BaseField {
    var out: BaseField;
    const slack = 2u;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = ((a.limbs[i + N] << slack) + (a.limbs[i + N - 1] >> (W - slack))) & W_mask;
    }
    return out;
}

// once reduces once (assumes that 0 <= a < 2 * mod)
fn field_reduce(a: BigInt256) -> BaseField {
    var res: BigInt256;
    var underflow = sub(a, BASE_MODULUS, &res);
    if (underflow == 1u) {
        return a;
    } else {
        return res;
    }
}

fn shorten(a: BigInt272) -> BigInt256 {
    var out: BigInt256;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = a.limbs[i];
    }
    return out;
}

// reduces l times (assumes that 0 <= a < multi * mod)
fn field_reduce_272(a: BigInt272, multi: u32) -> BaseField {
    var res: BigInt272;
    var cur = a;
    var cur_multi = multi + 1;
    while (cur_multi > 0u) {
        var underflow = sub_272(cur, BASE_MODULUS_MEDIUM_WIDE, &res);
        if (underflow == 1u) {
            return shorten(cur);
        } else {
            cur = res;
        }
        cur_multi = cur_multi - 1u;
    }
    return ZERO;
}

fn field_add(a: BaseField, b: BaseField) -> BaseField { 
    var res: BaseField;
    add(a, b, &res);
    return field_reduce(res);
}

fn field_sub(a: BaseField, b: BaseField) -> BaseField {
    var res: BaseField;
    var carry = sub(a, b, &res);
    if (carry == 0u) {
        return res;
    }
    add(res, BASE_MODULUS, &res);
    return res;
}

fn field_mul(a: BaseField, b: BaseField) -> BaseField {
    var xy: BigInt512 = mul(a, b);
    var xy_hi: BaseField = get_higher_with_slack(xy);
    var l: BigInt512 = mul(xy_hi, BASE_M);
    var l_hi: BaseField = get_higher_with_slack(l);
    var lp: BigInt512 = mul(l_hi, BASE_MODULUS);
    var r_wide: BigInt512;
    sub_512(xy, lp, &r_wide);

    var r_wide_reduced: BigInt512;
    var underflow = sub_512(r_wide, BASE_MODULUS_WIDE, &r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BaseField;
    for (var i = 0u; i < N; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return field_reduce(r);
}

// This is slow, probably don't want to use this
// fn field_small_scalar_mul(a: u32, b: BaseField) -> BaseField {
//     var constant: BaseField;
//     constant.limbs[0] = a;
//     return field_mul(constant, b);
// }

fn field_small_scalar_shift(l: u32, a: BaseField) -> BaseField { // max shift allowed is 16
    // assert (l < 16u);
    var res: BigInt272;
    for (var i = 0u; i < N; i = i + 1u) {
        let shift = a.limbs[i] << l;
        res.limbs[i] = res.limbs[i] | (shift & W_mask);
        res.limbs[i + 1] = (shift >> W);
    }

    var output = field_reduce_272(res, (1u << l)); // can probably be optimised
    return output;
}

fn field_pow(p: BaseField, e: u32) -> BaseField {
    var res: BaseField = p;
    for (var i = 1u; i < e; i = i + 1u) {
        res = field_mul(res, p);
    }
    return res;
}

fn field_eq(a: BaseField, b: BaseField) -> bool {
    for (var i = 0u; i < N; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) {
            return false;
        }
    }
    return true;
}

fn field_sqr(a: BaseField) -> BaseField {
    var xy: BigInt512 = sqr(a);
    var xy_hi: BaseField = get_higher_with_slack(xy);
    var l: BigInt512 = mul(xy_hi, BASE_M);
    var l_hi: BaseField = get_higher_with_slack(l);
    var lp: BigInt512 = mul(l_hi, BASE_MODULUS);
    var r_wide: BigInt512;
    sub_512(xy, lp, &r_wide);

    var r_wide_reduced: BigInt512;
    var underflow = sub_512(r_wide, BASE_MODULUS_WIDE, &r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BaseField;
    for (var i = 0u; i < N; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return field_reduce(r);
}

/*
fn field_to_bits(a: BigInt256) -> array<bool, 256> {
  let res: array<bool, 256> = array();
  for (var i = 0u;i < N;i += 1) {
    for (var j = 0u;j < 32u;j += 1) {
      var bit = (a.limbs[i] >> j) & 1u;
      res[i * 32u + j] = bit == 1u;
    }
  }
  return res;
}
*/


// curve.wgsl

struct JacobianPoint {
    x: BaseField,
    y: BaseField,
    z: BaseField
};

const JACOBIAN_IDENTITY: JacobianPoint = JacobianPoint(ZERO, ONE, ZERO);

fn is_inf(p: JacobianPoint) -> bool {
    return field_eq(p.z, ZERO);
}

fn jacobian_double(p: JacobianPoint) -> JacobianPoint {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    let A = field_sqr(p.x);
    let B = field_sqr(p.y);
    let C = field_sqr(B);
    let X1plusB = field_add(p.x, B);
    let D = field_small_scalar_shift(1, field_sub(field_sqr(X1plusB), field_add(A, C)));
    let E = field_add(field_small_scalar_shift(1, A), A);
    let F = field_sqr(E);
    let x3 = field_sub(F, field_small_scalar_shift(1, D));
    let y3 = field_sub(field_mul(E, field_sub(D, x3)), field_small_scalar_shift(3, C));
    let z3 = field_mul(field_small_scalar_shift(1, p.y), p.z);
    return JacobianPoint(x3, y3, z3);
}

// double p and add q
// todo: can be optimized if one of the z coordinates is 1
// fn jacobian_dadd(p: JacobianPoint, q: JacobianPoint) -> JacobianPoint {
//     if (is_inf(p)) {
//         return q;
//     } else if (is_inf(q)) {
//         return jacobian_double(p);
//     }

//     let twox = field_small_scalar_shift(1, p.x);
//     let sqrx = field_mul(p.x, p.x);
//     let dblR = field_add(field_small_scalar_shift(1, sqrx), sqrx);
//     let dblH = field_small_scalar_shift(1, p.y);

//     let x3 = field_mul(q.z, q.z);
//     let z3 = field_mul(p.z, q.z);
//     let addH = field_mul(p.z, p.z);

// }

fn jacobian_add(p: JacobianPoint, q: JacobianPoint) -> JacobianPoint {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
    if (field_eq(p.y, ZERO)) {
        return q;
    }
    if (field_eq(q.y, ZERO)) {
        return p;
    }

    let Z1Z1 = field_sqr(p.z);
    let Z2Z2 = field_sqr(q.z);
    let U1 = field_mul(p.x, Z2Z2);
    let U2 = field_mul(q.x, Z1Z1);
    let S1 = field_mul(p.y, field_mul(Z2Z2, q.z));
    let S2 = field_mul(q.y, field_mul(Z1Z1, p.z));
    if (field_eq(U1, U2)) {
        if (field_eq(S1, S2)) {
            return jacobian_double(p);
        } else {
            return JACOBIAN_IDENTITY;
        }
    }

    let H = field_sub(U2, U1);
    let I = field_small_scalar_shift(2, field_sqr(H));
    let J = field_mul(H, I);
    let R = field_small_scalar_shift(1, field_sub(S2, S1));
    let V = field_mul(U1, I);
    let nx = field_sub(field_sqr(R), field_add(J, field_small_scalar_shift(1, V)));
    let ny = field_sub(field_mul(R, field_sub(V, nx)), field_small_scalar_shift(1, field_mul(S1, J)));
    let nz = field_mul(H, field_sub(field_pow(field_add(p.z, q.z), 2), field_add(Z1Z1, Z2Z2)));
    return JacobianPoint(nx, ny, nz);
}

fn jacobian_mul(p: JacobianPoint, k: ScalarField) -> JacobianPoint {
    var r: JacobianPoint = JACOBIAN_IDENTITY;
    var t: JacobianPoint = p;
    for (var i = 0u; i < N; i = i + 1u) {
        var k_s = k.limbs[i];
        for (var j = 0u; j < W; j = j + 1u) {
            if ((k_s & 1) == 1u) {
                r = jacobian_add(r, t);
            }
            t = jacobian_double(t);
            k_s = k_s >> 1;
        }
    }
    return r;
}


// storage.wgsl

const WORKGROUP_SIZE = 64u;
const NUM_INVOCATIONS = 1u; // 4096u;
const MSM_SIZE = WORKGROUP_SIZE * NUM_INVOCATIONS;

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: array<JacobianPoint, NUM_INVOCATIONS>;
@group(0) @binding(3)
var<storage, read_write> mem: array<JacobianPoint, MSM_SIZE>;


// pippenger.wgsl

const POINTS_PER_INVOCATION = 64u;
const PARTITION_SIZE = 8u;
const POW_PART = (1u << PARTITION_SIZE);
const NUM_PARTITIONS = POINTS_PER_INVOCATION / PARTITION_SIZE; // 64 / 8 = 8
const PS_SZ = POW_PART; // 2^8 = 256
const BB_SIZE = 256;
const BB_SIZE_FAKE = 20;
const PS_SZ_NUM_INVOCATIONS = PS_SZ;// 1048576u; // PS_SZ * NUM_INVOCATIONS; // 256 * 4096 = 1048576
const BB_SIZE_NUM_INVOCATIONS = BB_SIZE;//1048576u; // BB_SIZE * NUM_INVOCATIONS; // 256 * 4096 = 1048576

@group(0) @binding(4)
var<storage, read_write> powerset_sums: array<JacobianPoint, PS_SZ_NUM_INVOCATIONS>;
@group(0) @binding(5)
var<storage, read_write> cur_sum: array<JacobianPoint, BB_SIZE_NUM_INVOCATIONS>;


@group(0) @binding(6)
var<storage, read_write> msm_len: u32;

fn pippenger(gidx: u32) -> JacobianPoint {
    var ps_base = gidx * PS_SZ;
    var sum_base = i32(gidx) * BB_SIZE;
    var point_base = gidx * POINTS_PER_INVOCATION;

    // first calculate power set sums for each partition of points
    // then calculate the sets for each point

    for(var bb = 0; bb < BB_SIZE; bb = bb + 1) {
        cur_sum[sum_base + bb] = JACOBIAN_IDENTITY;
    }
    for(var i = 0u; i < PS_SZ; i = i + 1) {
        powerset_sums[ps_base + i] = JACOBIAN_IDENTITY;
    }


    for(var i = 0u; i < NUM_PARTITIONS; i = i + 1) {

        // compute all power sums in this partition
        var idx = 0u;
        for(var j = 1u; j < POW_PART; j = j + 1){
            if((i32(j) & -i32(j)) == i32(j)) {
                powerset_sums[ps_base + j] = points[point_base + i * PARTITION_SIZE + idx];
                idx = idx + 1;
            } else {
                let cur_point = points[point_base + i * PARTITION_SIZE + idx];
                let mask = j & u32(j - 1);
                let other_mask = j ^ mask;
                powerset_sums[ps_base + j] = jacobian_add(powerset_sums[ps_base + mask], powerset_sums[ps_base + u32(other_mask)]);
            }
        }

        for(var bb: i32 = BB_SIZE - 1; bb >= 0; bb = bb - 1){
            var b = u32(bb);
            
            var powerset_idx = 0u;
            let modbW = b % W;
            let quotbW = b / W;
            for(var j = 0u; j < PARTITION_SIZE; j = j + 1){
                if((scalars[point_base + i * PARTITION_SIZE + j].limbs[quotbW] & (1u << modbW)) > 0) {
                    powerset_idx = powerset_idx | (1u << j);
                }
            }
            cur_sum[sum_base + bb] = jacobian_add(cur_sum[sum_base + bb], powerset_sums[ps_base + powerset_idx]);
        }
    }
    var running_total: JacobianPoint = JACOBIAN_IDENTITY;
    for(var bb = BB_SIZE - 1; bb >= 0; bb = bb - 1){
        running_total = jacobian_add(jacobian_double(running_total), cur_sum[sum_base + bb]);
    }
    return running_total;
}


// main.wgsl

// 3 -> 2
// storage -> workgroup
// second stage


@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    // result[gidx] = pippenger(gidx);
    // result[gidx] = jacobian_mul(points[gidx], scalars[gidx]);
    // result[0] = points[0];
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
