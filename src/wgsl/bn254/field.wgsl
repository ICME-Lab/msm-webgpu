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
    array(2788u, 40460u, 53156u, 3965u, 54731u, 24120u, 21946u, 41466u, 40585u, 63994u, 59685u, 7870u, 32601u, 31545u, 50740u, 15982u)
);

// Since W = 16 and BASE_MODULUS.limbs[0] = 64839,
//   we want n0' such that n0' * BASE_MODULUS.limbs[0] ≡ -1 mod 65536.
//   i.e. (n0' * 64839) & 0xFFFF = 0xFFFF.
// -(p^(-1)) % 2^16
const MONTGOMERY_INV = 25481u;

const ZERO: BigInt256 = BigInt256(
    array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

const ONE: BigInt256 = BigInt256(
    array(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)
);

fn get_higher_with_slack(a: BigInt512) -> BaseField {
    var out: BaseField;
    const slack = 2u; // 256 - 254
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

// This performs the "Montgomery reduce" on a 512-bit intermediate t.
// It returns t * R^{-1} mod M, provided 0 <= t < 2^256 * M.
fn montgomery_reduce(t: BigInt512) -> BaseField {
    var ret: BigInt512;
    // copy t into ret
    for (var i = 0u; i < 2u*N; i = i + 1u) {
        ret.limbs[i] = t.limbs[i];
    }

    // Outer loop: for each of the low 16 limbs
    for (var i = 0u; i < N; i = i + 1u) {
        let u = (ret.limbs[i] * MONTGOMERY_INV) & W_mask;

        var carry: u32 = 0u;
        for (var j = 0u; j < N; j = j + 1u) {
            let sum = ret.limbs[i + j] + (u * BASE_MODULUS.limbs[j]) + carry;
            ret.limbs[i + j] = sum & W_mask;
            carry = sum >> W;
        }
        ret.limbs[i + N] = ret.limbs[i + N] + carry;
    }

    // The result is in the high half ret.limbs[N..2N].
    var out: BigInt256;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = ret.limbs[i + N];
    }

    // Possibly subtract the modulus one time
    var tmp: BigInt256;
    let borrow = sub(out, BASE_MODULUS, &tmp);
    if (borrow == 0u) {
        // out >= M, so out = out - M
        out = tmp;
    }

    return out;
}

// Multiplies two 256-bit field elements (in normal form) using Montgomery reduction.
// If you maintain your elements in Montgomery form, you would first convert them 
// from Montgomery to “normal” or vice versa as needed. For a simple demonstration
// where a and b are in normal form, we do a plain 256x256->512 multiply, then reduce.
fn field_mul(a: BaseField, b: BaseField) -> BaseField {
    // 1) Multiply into 512 bits:
    let t: BigInt512 = mul(a, b);

    // 2) Montgomery-reduce the 512-bit product down to 256 bits:
    let result = montgomery_reduce(t);

    return result;
}


//------------------------------------------------------
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
    var res = field_mul(a, a);
    return res;
}
