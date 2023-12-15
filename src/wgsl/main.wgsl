const W = 16u;
const W_mask = 65535u;
const L = 256;
const N = 16u;
const N2 = 32u;
const NPlusOne = 17u;

// No overflow
struct BigInt256 {
    limbs: array<u32,N>
}

struct BigInt512 {
    limbs: array<u32,N2>
}

struct BigInt272 {
    limbs: array<u32,NPlusOne>
}

fn add(a: ptr<function, BigInt256>, b: ptr<function, BigInt256>, res: ptr<function, BigInt256>) -> u32 {
    var carry: u32 = 0u;
    for (var j: u32 = 0u; j < 16u; j ++) {
        let c: u32 = (*a).limbs[j] + (*b).limbs[j] + carry;
        (*res).limbs[j] = c & 65535u;
        carry = c >> 16u;
    }
    return carry;
}

fn sub(a: ptr<function, BigInt256>, b: ptr<function, BigInt256>, res: ptr<function, BigInt256>) -> u32 {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        (*res).limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            (*res).limbs[i] += 65536u;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn sub_512(a: ptr<function, BigInt512>, b: ptr<function, BigInt512>, res: ptr<function, BigInt512>) -> u32 {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < (N2); i = i + 1u) {
        (*res).limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1u;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

// assumes a >= b
fn sub_272(a: ptr<function, BigInt272>, b: ptr<function, BigInt272>, res: ptr<function, BigInt272>) -> u32 {
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < NPlusOne; i = i + 1u) {
        (*res).limbs[i] = (*a).limbs[i] - (*b).limbs[i] - borrow;
        if ((*a).limbs[i] < ((*b).limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1u;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn mul(a: ptr<function, BigInt256>, b: ptr<function, BigInt256>) -> BigInt512 {
    var res: BigInt512;
    let N = 16u;
    for (var i = 0u; i < N; i = i + 1u) {
        for (var j = 0u; j < N; j = j + 1u) {
            let c = (*a).limbs[i] * (*b).limbs[j];
            res.limbs[i+j] += c & ((1u << 16u) - 1u);
            res.limbs[i+j+1u] += c >> 16u;
        }   
    }
    // start from 0 and carry the extra over to the next index
    for (var i = 0u; i < (N2 - 1u); i = i + 1u) {
        res.limbs[i+1u] += res.limbs[i] >> 16u;
        res.limbs[i] = res.limbs[i] & ((1u << 16u) - 1u);
    }
    return res;
}

alias BaseField = BigInt256;
alias ScalarField = BigInt256;

const BASE_NBITS = 255;

fn get_base_mod() -> BigInt256 {
    var p: BigInt256;
    p.limbs[0] = 1u;
    p.limbs[1] = 0u;
    p.limbs[2] = 12525u;
    p.limbs[3] = 39213u;
    p.limbs[4] = 63771u;
    p.limbs[5] = 2380u;
    p.limbs[6] = 39164u;
    p.limbs[7] = 8774u;
    p.limbs[8] = 0u;
    p.limbs[9] = 0u;
    p.limbs[10] = 0u;
    p.limbs[11] = 0u;
    p.limbs[12] = 0u;
    p.limbs[13] = 0u;
    p.limbs[14] = 0u;
    p.limbs[15] = 16384u;
    return p;
}

fn get_base_mod_med_wide() -> BigInt272 {
    var p: BigInt272;
    p.limbs[0] = 1u;
    p.limbs[1] = 0u;
    p.limbs[2] = 12525u;
    p.limbs[3] = 39213u;
    p.limbs[4] = 63771u;
    p.limbs[5] = 2380u;
    p.limbs[6] = 39164u;
    p.limbs[7] = 8774u;
    p.limbs[8] = 0u;
    p.limbs[9] = 0u;
    p.limbs[10] = 0u;
    p.limbs[11] = 0u;
    p.limbs[12] = 0u;
    p.limbs[13] = 0u;
    p.limbs[14] = 0u;
    p.limbs[15] = 16384u;
    p.limbs[16] = 0u;
    return p;
}

fn get_base_mod_wide() -> BigInt512 {
    var p: BigInt512;
    p.limbs[0] = 1u;
    p.limbs[1] = 0u;
    p.limbs[2] = 12525u;
    p.limbs[3] = 39213u;
    p.limbs[4] = 63771u;
    p.limbs[5] = 2380u;
    p.limbs[6] = 39164u;
    p.limbs[7] = 8774u;
    p.limbs[8] = 0u;
    p.limbs[9] = 0u;
    p.limbs[10] = 0u;
    p.limbs[11] = 0u;
    p.limbs[12] = 0u;
    p.limbs[13] = 0u;
    p.limbs[14] = 0u;
    p.limbs[15] = 16384u;
    p.limbs[16] = 0u;
    p.limbs[17] = 0u;
    p.limbs[18] = 0u;
    p.limbs[19] = 0u;
    p.limbs[20] = 0u;
    p.limbs[21] = 0u;
    p.limbs[22] = 0u;
    p.limbs[23] = 0u;
    p.limbs[24] = 0u;
    p.limbs[25] = 0u;
    p.limbs[26] = 0u;
    p.limbs[27] = 0u;
    p.limbs[28] = 0u;
    p.limbs[29] = 0u;
    p.limbs[30] = 0u;
    p.limbs[31] = 0u;
    return p;
}

fn get_base_m() -> BigInt256 {
    var p: BigInt256;
    p.limbs[0] = 65532u;
    p.limbs[1] = 65535u;
    p.limbs[2] = 15435u;
    p.limbs[3] = 39755u;
    p.limbs[4] = 7057u;
    p.limbs[5] = 56012u;
    p.limbs[6] = 39951u;
    p.limbs[7] = 30437u;
    p.limbs[8] = 65535u;
    p.limbs[9] = 65535u;
    p.limbs[10] = 65535u;
    p.limbs[11] = 65535u;
    p.limbs[12] = 65535u;
    p.limbs[13] = 65535u;
    p.limbs[14] = 65535u;
    p.limbs[15] = 65535u;
    return p;
}

fn get_zero() -> BigInt256 {
    var p: BigInt256;
    p.limbs[0] = 0u;
    p.limbs[1] = 0u;
    p.limbs[2] = 0u;
    p.limbs[3] = 0u;
    p.limbs[4] = 0u;
    p.limbs[5] = 0u;
    p.limbs[6] = 0u;
    p.limbs[7] = 0u;
    p.limbs[8] = 0u;
    p.limbs[9] = 0u;
    p.limbs[10] = 0u;
    p.limbs[11] = 0u;
    p.limbs[12] = 0u;
    p.limbs[13] = 0u;
    p.limbs[14] = 0u;
    p.limbs[15] = 0u;
    return p;
}

fn get_one() -> BigInt256 {
    var p: BigInt256;
    p.limbs[0] = 1u;
    p.limbs[1] = 0u;
    p.limbs[2] = 0u;
    p.limbs[3] = 0u;
    p.limbs[4] = 0u;
    p.limbs[5] = 0u;
    p.limbs[6] = 0u;
    p.limbs[7] = 0u;
    p.limbs[8] = 0u;
    p.limbs[9] = 0u;
    p.limbs[10] = 0u;
    p.limbs[11] = 0u;
    p.limbs[12] = 0u;
    p.limbs[13] = 0u;
    p.limbs[14] = 0u;
    p.limbs[15] = 0u;
    return p;
}

fn get_higher_with_slack(a: ptr<function, BigInt512>) -> BaseField {
    var out: BaseField;
    var slack = 256u - 255u;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = (((*a).limbs[i + N] << slack) + ((*a).limbs[i + N - 1u] >> (W - slack))) & W_mask;
    }
    return out;
}

// once reduces once (assumes that 0 <= a < 2 * mod)
fn field_reduce(a: ptr<function, BigInt256>) -> BaseField {
    var res: BigInt256;
    var base_mod: BigInt256 = get_base_mod();
    var underflow = sub(a, &base_mod, &res);
    if (underflow == 1u) {
        return *a;
    } else {
        return res;
    }
}

fn shorten(a: ptr<function, BigInt272>) -> BigInt256 {
    var out: BigInt256;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = (*a).limbs[i];
    }
    return out;
}

// reduces l times (assumes that 0 <= a < multi * mod)
fn field_reduce_272(a: ptr<function, BigInt272>, multi: u32) -> BaseField {
    var zero = get_zero();
    var ans = zero;
    var res: BigInt272;
    var cur = *a;
    var cur_multi = multi + 1u;
    var mod_med_wide = get_base_mod_med_wide();
    while (cur_multi > 0u) {
        var underflow = sub_272(&cur, &mod_med_wide, &res);
        if (underflow == 1u) {
            ans = shorten(&cur);

            //return statement was corrupting output for some reason?
            cur_multi = 1u;
        } else {
            cur = res;
        }
        cur_multi = cur_multi - 1u;
    }

    return ans;
}

fn field_add(a: ptr<function, BaseField>, b: ptr<function, BaseField>) -> BaseField { 
    var res: BaseField;
    add(a, b, &res);
    return field_reduce(&res);
}

fn field_sub(a: ptr<function, BaseField>, b: ptr<function, BaseField>) -> BaseField {
    var res: BaseField;
    var carry = sub(a, b, &res);
    if (carry == 0u) {
        return res;
    }
    var base_mod: BigInt256 = get_base_mod();
    var other_res = res;
    add(&other_res, &base_mod, &res);
    return res;
}

fn field_mul(a: ptr<function, BaseField>, b: ptr<function, BaseField>) -> BaseField {
    var xy: BigInt512 = mul(a, b);
    var xy_hi: BaseField = get_higher_with_slack(&xy);
    var base_m: BigInt256 = get_base_m();
    var l: BigInt512 = mul(&xy_hi, &base_m);
    var l_hi: BaseField = get_higher_with_slack(&l);
    var base_mod: BigInt256 = get_base_mod();
    var lp: BigInt512 = mul(&l_hi, &base_mod);
    var r_wide: BigInt512;
    sub_512(&xy, &lp, &r_wide);

    var r_wide_reduced: BigInt512;
    var base_mod_wide: BigInt512 = get_base_mod_wide();
    var underflow = sub_512(&r_wide, &base_mod_wide, &r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BaseField;
    for (var i = 0u; i < N; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return field_reduce(&r);
}

fn field_small_scalar_mul(a: u32, b: ptr<function, BaseField>) -> BaseField {
    var constant: BaseField;
    constant.limbs[0] = a;
    return field_mul(&constant, b);
}

fn field_small_scalar_shift(l: u32, a: ptr<function, BaseField>) -> BaseField { // max shift allowed is 16
    // assert (l < 16u);
    var res: BigInt272;
    for (var i = 0u; i < N; i = i + 1u) {
        let shift = (*a).limbs[i] << l;
        res.limbs[i] = res.limbs[i] | (shift & W_mask);
        res.limbs[i + 1u] = (shift >> W);
    }

    var output = field_reduce_272(&res, (1u << l)); // can probably be optimised
    return output;
}

fn field_pow(p: ptr<function, BaseField>, e: u32) -> BaseField {
    var res: BaseField = *p;
    for (var i = 1u; i < e; i = i + 1u) {
        res = field_mul(&res, p);
    }
    return res;
}

fn field_eq(a: ptr<function, BaseField>, b: ptr<function, BaseField>) -> bool {
    for (var i = 0u; i < N; i = i + 1u) {
        if ((*a).limbs[i] != (*b).limbs[i]) {
            return false;
        }
    }
    return true;
}

/*
    -----------------------------------------------CURVES----------------------------------------------------------
*/
struct JacobianPoint {
    x: BaseField,
    y: BaseField,
    z: BaseField
};

fn is_inf(p: ptr<function, JacobianPoint>) -> bool {
    var pz = (*p).z;
    var zero: BigInt256 = get_zero();
    return field_eq(&pz, &zero);
}

fn slow_jacobian_double(p: ptr<function, JacobianPoint>) -> JacobianPoint {
    var zero: BigInt256 = get_zero();
    var py = (*p).y;
    var px = (*p).x;
    var pz = (*p).z;

    if (field_eq(&py, &zero)) {
        return JacobianPoint(zero, zero, zero);
    }
    
    var ysq: BigInt256 = field_pow(&py, 2u);
    var s1: BigInt256 = field_mul(&px, &ysq);
    var m1: BigInt256 = field_pow(&px, 2u);

    var S: BigInt256 = field_small_scalar_mul(4u, &s1);
    var M: BigInt256 = field_small_scalar_mul(3u, &m1); // assumes a = 0, sw curve

    //nx field sub args
    var nx1 = field_pow(&M, 2u);
    var nx2 = field_small_scalar_mul(2u, &S);

    var nx = field_sub(&nx1, &nx2);
    
    //ny field sub args
    var ny1_inner_fs = field_sub(&S, &nx);
    var ny1 = field_mul(&M, &ny1_inner_fs);

    var ny2_inner_fp = field_pow(&ysq, 2u);
    var ny2 = field_small_scalar_mul(8u, &ny2_inner_fp);

    var ny = field_sub(&ny1, &ny2);

    //nz feild mul args
    var nz1 = field_small_scalar_mul(2u, &py);
    var nz = field_mul(&nz1, &pz);

    return JacobianPoint(nx, ny, nz);
}

fn jacobian_double(p: ptr<function, JacobianPoint>) -> JacobianPoint {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    var zero: BigInt256 = get_zero();
    var py = (*p).y;
    var px = (*p).x;
    var pz = (*p).z;
    // if (field_eq(&py, &zero)) {
    //     return JacobianPoint(zero, zero, zero);
    // }

    var A = field_mul(&px, &px);
    var B = field_mul(&py, &py);
    var C = field_mul(&B, &B);

    var X1plusB = field_add(&px, &B);

    var d_fs_1 = field_mul(&X1plusB, &X1plusB);
    var d_fs_2 = field_add(&A, &C);
    var d_2 = field_sub(&d_fs_1, &d_fs_2);
    var D = field_small_scalar_shift(1u, &d_2);

    var e_1 = field_small_scalar_shift(1u, &A);
    var E = field_add(&e_1, &A);
    var F = field_mul(&E, &E);

    var x3_2 = field_small_scalar_shift(1u, &D);
    var x3 = field_sub(&F, &x3_2);

    var y3_1_fs = field_sub(&D, &x3);
    var y3_1 = field_mul(&E, &y3_1_fs);
    var y3_2 = field_small_scalar_shift(3u, &C);
    var y3 = field_sub(&y3_1, &y3_2);

    var z3_1 = field_small_scalar_shift(1u, &py);
    var z3 = field_mul(&z3_1, &pz);
    var one = get_one();

    return JacobianPoint(x3, y3, z3);
}

fn jacobian_add(p: ptr<function, JacobianPoint>, q: ptr<function, JacobianPoint>) -> JacobianPoint {
    var zero = get_zero();

    var px = (*p).x;
    var py = (*p).y;
    var pz = (*p).z;

    var qx = (*q).x;
    var qy = (*q).y;
    var qz = (*q).z;

    if (field_eq(&py, &zero)) {
        return *q;
    }
    if (field_eq(&qy, &zero)) {
        return *p;
    }

    var Z1Z1 = field_mul(&pz, &pz);
    var Z2Z2 = field_mul(&qz, &qz);
    var U1 = field_mul(&px, &Z1Z1);
    var U2 = field_mul(&qx, &Z2Z2);

    var s1_2 = field_mul(&Z2Z2, &qz);
    var S1 = field_mul(&py, &s1_2);

    var s2_2 = field_mul(&Z1Z1, &pz);
    var S2 = field_mul(&qy, &s2_2);

    if (field_eq(&U1, &U2)) {
        if (field_eq(&S1, &S2)) {
            var j_double = jacobian_double(p);
            return j_double;
        } else {
            return JacobianPoint(zero, zero, zero);
        }
    }

    var H = field_sub(&U2, &U1);
    
    var i_2 = field_mul(&H, &H);
    var I = field_small_scalar_shift(2u, &i_2);
    var J = field_mul(&H, &I);

    var r_2 = field_sub(&S2, &S1);
    var R = field_small_scalar_shift(1u, &r_2);
    var V = field_mul(&U1, &I);

    var nx_1 = field_mul(&R, &R);

    var nx_2_fa_2 = field_small_scalar_shift(1u, &V);
    var nx_2 = field_add(&J, &nx_2_fa_2);
    var nx = field_sub(&nx_1, &nx_2);

    var ny_1_fm_2 = field_sub(&V, &nx);
    var ny_1 = field_mul(&R, &ny_1_fm_2);

    var ny_2_fss_2 = field_mul(&S1, &J);
    var ny_2 = field_small_scalar_shift(1u, &ny_2_fss_2);

    var ny = field_sub(&ny_1, &ny_2);

    var nz_2_fs_1_fa = field_add(&pz, &qz);
    var nz_2_fs_1 = field_pow(&nz_2_fs_1_fa, 2u);
    var nz_2_fs_2 = field_add(&Z1Z1, &Z2Z2);
    var nz_2 = field_sub(&nz_2_fs_1, &nz_2_fs_2);
    var nz = field_mul(&H, &nz_2);
    
    return JacobianPoint(nx, ny, nz);
}

fn jacobian_mul(p: ptr<function, JacobianPoint>, k: ptr<function, ScalarField>) -> JacobianPoint {

    var zero = get_zero();
    var r: JacobianPoint = JacobianPoint(zero, zero, zero);
    var t: JacobianPoint = *p;

    for (var i = 0u; i < N; i = i + 1u) {
        var k_s = (*k).limbs[i];
        for (var j = 0u; j < W; j = j + 1u) {
            if ((k_s & 1u) == 1u) {
                r = jacobian_add(&r, &t);
            }
            t = jacobian_double(&t);
            k_s = k_s >> 1u;
        }
    }
    return r;
}

struct Array {
	data: array<u32>
};

@group(0) @binding(0)
var<storage, read_write> points: array<JacobianPoint>;
@group(0) @binding(1)
var<storage, read_write> scalars: array<ScalarField>;
@group(0) @binding(2)
var<storage, read_write> result: JacobianPoint;

const WORKGROUP_SIZE = 1u;
var<workgroup> mem: array<JacobianPoint, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let gidx = global_id.x;
    let lidx = local_id.x;

    var point = points[lidx];
    var scalar = scalars[lidx];
    mem[gidx] = jacobian_mul(&point, &scalar);
    
    workgroupBarrier();
    for (var offset: u32 = WORKGROUP_SIZE / 2u; offset > 0u; offset = offset / 2u) {
        if (lidx < offset) {
            var mem_lidx = mem[lidx];
            var mem_lidx_off = mem[lidx + offset];
            mem[lidx] = jacobian_add(&mem_lidx, &mem_lidx_off);
        }
    workgroupBarrier();

    }

    // // // TDO: read about memory ordering and fix this when we have multiple global invocations
    if (lidx == 0u) {
        result = mem[0];
    }
}