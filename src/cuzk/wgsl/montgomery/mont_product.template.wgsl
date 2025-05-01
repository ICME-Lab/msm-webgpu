
// This performs the "Montgomery reduce" on a 512-bit intermediate t.
// It returns t * R^{-1} mod M, provided 0 <= t < 2^256 * M.
fn montgomery_reduce_2(t: ptr<function, BigIntWide>) -> BigInt {
    var ret: BigIntWide;
    var p = get_p();
    // copy t into ret
    for (var i = 0u; i < 2u*NUM_WORDS; i = i + 1u) {
        ret.limbs[i] = (*t).limbs[i];
    }

    // Outer loop: for each of the low 16 limbs
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        let u = (ret.limbs[i] * N0) & W_MASK;

        var carry: u32 = 0u;
        for (var j = 0u; j < NUM_WORDS; j = j + 1u) {
            let sum = ret.limbs[i + j] + (u * p.limbs[j]) + carry;
            ret.limbs[i + j] = sum & W_MASK;
            carry = sum >> WORD_SIZE;
        }
        ret.limbs[i + NUM_WORDS] = ret.limbs[i + NUM_WORDS] + carry;
    }

    // The result is in the high half ret.limbs[N..2N].
    var out: BigInt;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        out.limbs[i] = ret.limbs[i + NUM_WORDS];
    }

    // Possibly subtract the modulus one time
    var tmp: BigInt;
    let borrow = bigint_sub(&out, &p, &tmp);
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
fn montgomery_product_2(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> BigInt {
    // 1) Multiply into 512 bits:
    var t: BigIntWide = bigint_mul(a, b);

    // 2) Montgomery-reduce the 512-bit product down to 256 bits:
    let result = montgomery_reduce_2(&t);

    return result;
}

fn montgomery_square_2(x: ptr<function, BigInt>) -> BigInt {
    return montgomery_product_2(x, x);
}

fn montgomery_pow_2(p: ptr<function, BigInt>, e: u32) -> BigInt {
    var res: BigInt = *p;
    for (var i = 1u; i < e; i = i + 1u) {
        res = montgomery_product_2(&res, p);
    }
    return res;
}