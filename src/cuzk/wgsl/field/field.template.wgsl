const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;
const W_MASK = {{ w_mask }}u;
const N0 = {{ n0 }}u;

const ZERO: BigInt = BigInt(
    array({{ zero_limbs }})
);

fn get_p() -> BigInt {
    var p: BigInt;
{{{ p_limbs }}}
    return p;
}

fn get_p_limbs_plus_one() -> BigIntMediumWide {
    var p: BigIntMediumWide;
{{{ p_limbs_plus_one }}}
    return p;
}

fn get_r() -> BigInt {
    var r: BigInt;
{{{ r_limbs }}}
    return r;
}

fn get_rinv() -> BigInt {
    var rinv: BigInt;
{{{ rinv_limbs }}}
    return rinv;
}



fn field_add(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    bigint_add(a, b, &res);
    return field_reduce(&res);
}

fn field_reduce(a: ptr<function, BigInt>) -> BigInt {
    var res: BigInt;
    var p: BigInt = get_p();
    var underflow = bigint_sub(a, &p, &res);
    if (underflow == 1u) {
        return *a;
    }

    return res;
}

fn field_sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt { 
    var res: BigInt;
    
    var c = bigint_gt(a, b);
    /// a <= b.
    if (c == 0u) { 
        var r: BigInt;
        bigint_sub(b, a, &r);
        var p = get_p();
        bigint_sub(&p, &r, &res);
        return res;
    } 
    /// a > b.
    else { 
        bigint_sub(a, b, &res);
        return res;
    }
}

fn sub_medium_wide(a: BigIntMediumWide, b: BigIntMediumWide, res: ptr<function, BigIntMediumWide>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < NUM_WORDS + 1; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_MASK + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}


fn field_eq(a: BigInt, b: BigInt) -> bool {
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) {
            return false;
        }
    }
    return true;
}

fn shorten(a: BigIntMediumWide) -> BigInt {
    var out: BigInt;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        out.limbs[i] = a.limbs[i];
    }
    return out;
}

fn field_reduce_medium_wide(a: BigIntMediumWide, multi: u32) -> BigInt {
    var res: BigIntMediumWide;
    var cur = a;
    var cur_multi = multi + 1;
    while (cur_multi > 0u) {
        var base_modulus_medium_wide = get_p_limbs_plus_one();
        var underflow = sub_medium_wide(cur, base_modulus_medium_wide, &res);
        if (underflow == 1u) {
            return shorten(cur);
        } else {
            cur = res;
        }
        cur_multi = cur_multi - 1u;
    }
    return ZERO;
}

fn field_small_scalar_shift(l: u32, a: BigInt) -> BigInt { // max shift allowed is 16
    var res: BigIntMediumWide;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        let shift = a.limbs[i] << l;
        res.limbs[i] = res.limbs[i] | (shift & W_MASK);
        res.limbs[i + 1] = (shift >> WORD_SIZE);
    }

    var output = field_reduce_medium_wide(res, (1u << l)); // can probably be optimised
    return output;
}

