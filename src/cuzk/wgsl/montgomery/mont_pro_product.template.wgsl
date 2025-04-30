
const NUM_WORDS = {{ num_words }}u;
const WORD_SIZE = {{ word_size }}u;
const W_MASK = {{ w_mask }}u;
const N0 = {{ n0 }}u;



/// An optimised variant of the Montgomery product algorithm from
/// https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication.
fn montgomery_product(x: ptr<function, BigInt>, y: ptr<function, BigInt>) -> BigInt {
    var s: BigInt;
    var p = get_p();

    for (var i = 0u; i < NUM_WORDS; i ++) {
        var t = s.limbs[0] + (*x).limbs[i] * (*y).limbs[0];
        var tprime = t & W_MASK;
        var qi = (N0 * tprime) & W_MASK;
        var c = (t + qi * p.limbs[0]) >> WORD_SIZE;
        s.limbs[0] = s.limbs[1] + (*x).limbs[i] * (*y).limbs[1] + qi * p.limbs[1] + c;

        /// Since nSafe = 32 when NUM_WORDS = 20, we can perform the following
        /// iterations without performing a carry.
        for (var j = 2u; j < NUM_WORDS; j ++) {
            s.limbs[j - 1u] = s.limbs[j] + (*x).limbs[i] * (*y).limbs[j] + qi * p.limbs[j];
        }
        s.limbs[NUM_WORDS - 2u] = (*x).limbs[i] * (*y).limbs[NUM_WORDS - 1u] + qi * p.limbs[NUM_WORDS - 1u];
    }

    /// To paraphrase mitschabaude: a last round of carries to ensure that each
    /// limb is at most WORD_SIZE bits.
    var c = 0u;
    for (var i = 0u; i < NUM_WORDS; i ++) {
        var v = s.limbs[i] + c;
        c = v >> WORD_SIZE;
        s.limbs[i] = v & W_MASK; 
    }

    return conditional_reduce(&s, &p);
}

fn conditional_reduce(x: ptr<function, BigInt>, y: ptr<function, BigInt>) -> BigInt {
    /// Determine if x > y.
    var x_gt_y = bigint_gt(x, y);

    if (x_gt_y == 1u) {
        var res: BigInt;
        bigint_sub(x, y, &res);
        return res;
    }

    return *x;
}

fn montgomery_square(x: ptr<function, BigInt>) -> BigInt {
    return montgomery_product(x, x);
}

fn montgomery_pow(p: ptr<function, BigInt>, e: u32) -> BigInt {
    var res: BigInt = *p;
    for (var i = 1u; i < e; i = i + 1u) {
        res = montgomery_product(&res, p);
    }
    return res;
}