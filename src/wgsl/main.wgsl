const W = 16u;
const W_mask = 65535u;
const L = 256;
const N = 16u;
const N2 = 32u;

// No overflow
struct BigInt256 {
    limbs: array<u32,N>
}

struct BigInt512 {
    limbs: array<u32,N2>
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

const BASE_NBITS = 255;

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

const ONE: BigInt256 = BigInt256(
    array(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

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

fn jacobian_double(p: ptr<function, JacobianPoint>) -> JacobianPoint {
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

fn jacobian_add(p: ptr<function, JacobianPoint>, q: ptr<function, JacobianPoint>) -> JacobianPoint {
    var zero = get_zero();
    var py = (*p).y;
    var px = (*p).x;
    var pz = (*p).z;

    var qy = (*q).y;
    var qx = (*q).x;
    var qz = (*q).z;

    if (field_eq(&py, &zero)) {
        return *q;
    }
    if (field_eq(&qy, &zero)) {
        return *p;
    }

    var fp_qz_2 = field_pow(&qz, 2u);
    var fp_pz_2 = field_pow(&pz, 2u);
    var fp_qz_3 = field_pow(&qz, 3u);
    var fp_pz_3 = field_pow(&pz, 3u);

    var U1 = field_mul(&px, &fp_qz_2);
    var U2 = field_mul(&qx, &fp_pz_2);
    var S1 = field_mul(&py, &fp_qz_3);
    var S2 = field_mul(&qy, &fp_pz_3);

    if (field_eq(&U1, &U2)) {
        if (field_eq(&S1, &S2)) {
            var j_double = jacobian_double(p);
            return j_double;
        } else {
            return JacobianPoint(zero, zero, zero);
        }
    }

    var H = field_sub(&U2, &U1);
    var R = field_sub(&S2, &S1);
    var H2 = field_mul(&H, &H);
    var H3 = field_mul(&H2, &H);
    var U1H2 = field_mul(&U1, &H2);

    //nx args
    var nx1_fp_inner = field_pow(&R, 2u);
    var nx1 = field_sub(&nx1_fp_inner, &H3);
    var nx2 = field_small_scalar_mul(2u, &U1H2);
    var nx = field_sub(&nx1, &nx2);

    //ny args
    var ny1_fs_inner = field_sub(&U1H2, &nx);
    var ny1 = field_mul(&R, &ny1_fs_inner);
    var ny2 = field_mul(&S1, &H3);

    var ny = field_sub(&ny1, &ny2);

    var nz1 = field_mul(&H, &pz);
    var nz = field_mul(&nz1, &qz);
    
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

const WORKGROUP_SIZE = 128;

var<workgroup> mem: array<JacobianPoint, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var gidx = global_id.x;
    var lidx = local_id.x;

    var point_1 = points[gidx];
    var scalar_1 = scalars[gidx];
    mem[lidx] = jacobian_mul(&point_1, &scalar_1);

    workgroupBarrier();
    
    for (var offset: u32 = 128u / 2u; offset > 0u; offset = offset / 2u) {
        if (lidx < offset) {
            var mem_1 = mem[lidx];
            var mem_2 = mem[lidx + offset];

            mem[lidx] = jacobian_add(&mem_1, &mem_2);
        }
        workgroupBarrier();
    }

    // TODO: read about memory ordering and fix this when we have multiple global invocations
    if (lidx == 0u) {
        result = mem[0];
    }
}