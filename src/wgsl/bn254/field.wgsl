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

fn is_zero(b: BaseField) -> bool {
    for (var i = 0u; i < N; i = i + 1u) {
        if (b.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

fn shift_right_by_1(b: BigInt256) -> BigInt256 {
    var res: BigInt256;
    var carry_in: u32 = 0;
    for (var i = N; i > 0u; /* decrement i in a safe way */) {
        i = i - 1u;
        let cur = b.limbs[i] | (carry_in << W);
        res.limbs[i] = cur >> 1;
        carry_in = cur & 1u; 
    }
    return res;
}

fn double(a: BigInt256) -> BigInt256 {
    // shift a by 1 bit.  Since each limb is 16 bits, do a 16-limb carry.
    var res: BigInt256;
    var carry: u32 = 0;
    for (var i = 0u; i < N; i = i + 1u) {
        let tmp = (a.limbs[i] << 1) + carry;
        res.limbs[i] = tmp & W_mask;
        carry = tmp >> W;
    }
    return res;
}

fn field_mul_naive(a: BaseField, b: BaseField) -> BaseField {
    var accumulator = ZERO;
    var newA = a;
    var newB = b;

    while (!is_zero(newB)) {
        // If newB is odd:
        if ((newB.limbs[0] & 1u) == 1u) {
            // accumulator += newA (mod M)
            add(accumulator, newA, &accumulator);
            // reduce if necessary
            var tmp: BigInt256;
            let borrow = sub(accumulator, BASE_MODULUS, &tmp);
            if (borrow == 0u) {
                accumulator = tmp;
            }
        }

        // newA = 2 * newA (mod M)
        newA = double(newA);
        // reduce once if needed
        var tmp2: BigInt256;
        let borrow2 = sub(newA, BASE_MODULUS, &tmp2);
        if (borrow2 == 0u) {
            newA = tmp2;
        }

        // newB = newB >> 1
        newB = shift_right_by_1(newB);
    }

    return accumulator;
}


// fn field_mul(a: BaseField, b: BaseField) -> BaseField {
//   // The “t” array and 2-element “t_extra” as in your snippet
//     var t = array<u32, 16>();
//     var t_extra = array<u32, 2>();
//     // Initialize everything to 0
//     for (var idx = 0u; idx < N; idx = idx + 1u) {
//         t[idx] = 0u;
//     }
//     t_extra[0] = 0u;
//     t_extra[1] = 0u;

//     var i = N; // We'll decrement from 15 down to 0
//     loop {
//         if (i == 0u) {
//             break;
//         }
//         i = i - 1u;

//         // 1) Multiply a by b[i], add into t, tracking carry
//         var c = 0u;
//         var j = N;
//         loop {
//             if (j == 0u) {
//                 break;
//             }
//             j = j - 1u;

//             let aj    = a.limbs[j];      // up to 16 bits
//             let bi    = b.limbs[i];      // up to 16 bits
//             let tj    = t[j];            // up to 16 bits stored, but in a u32
//             // sum = t[j] + (a[j]*b[i]) + carry
//             let partial = (aj * bi) + tj + c;
//             // Low 16 bits go to t[j], high 16 bits become new carry
//             t[j] = partial & W_mask;
//             c    = partial >> W; // up to 16 bits
//         }

//         // 2) Add leftover carry c into t_extra
//         //    t_extra[1] + c => store low 16 bits, the high bits go into t_extra[0]
//         let temp2  = t_extra[1] + c;
//         t_extra[1] = temp2 & W_mask;
//         t_extra[0] = (temp2 >> W); // can be 0 or 1, for instance

//         // 3) m = the low 32 bits of (t[NUM_LIMBS-1] * MU)
//         //    but t[15] is up to 16 bits, MU up to 16/32 bits => product fits in 32 bits
//         let top_limb = t[N - 1u]; // t[15]
//         let prod_mu  = top_limb * MU;     // up to 32 bits
//         let m        = prod_mu & 0xFFFFFFFFu;
//         // (In the C++ code, `(((uint64_t) x << 32) >> 32)` is basically x & 0xffffffff.)

//         // 4) c = (t[15] + m * MOD.limbs[15]) >> 16
//         let top_mod  = BASE_MODULUS.limbs[N - 1u]; // the top limb (16 bits)
//         let tmpSum   = t[N - 1u] + (m * top_mod);
//         var carryC   = tmpSum >> W; // high 16 bits

//         // 5) For j in [14 .. 0], do: cs = t[j] + m * MOD[j] + carry
//         //    c = cs >> 16, t[j+1] = cs & 0xffff
//         var kk = N - 1u; // 15
//         loop {
//             if (kk == 0u) {
//                 break;
//             }
//             kk = kk - 1u; // from 14 down to 0
//             let sumVal = t[kk] + (m * BASE_MODULUS.limbs[kk]) + carryC;
//             t[kk + 1u] = sumVal & W_mask;
//             carryC     = sumVal >> W;
//         }

//         // 6) Combine leftover carryC into t_extra and store final t[0]
//         //    cs = t_extra[1] + carryC => c = cs >>16, t[0] = cs & 0xffff
//         let sum6   = t_extra[1] + carryC;
//         let c2     = sum6 >> W;
//         t[0]       = sum6 & W_mask;
//         // Then t_extra[1] = t_extra[0] + c2
//         t_extra[1] = t_extra[0] + c2;  // up to 16 bits if c2 is small
//     }

//     // 7) Now package up t as the candidate result
//     var result: BaseField;
//     for (var idx = 0u; idx < N; idx = idx + 1u) {
//         result.limbs[idx] = t[idx];
//     }

//     // 8) Check overflow or if result >= modulus
//     //    overflow if t_extra[0] > 0 (the “high carry”)
//     let overflow = select(false, true, t_extra[0] > 0u);

//     // If we overflowed or result >= MOD, do result = result - MOD
//     if (overflow || !field_less_than(result, BASE_MODULUS)) {
//         var dummy = BaseField(array<u32,16>(
//             // initialize to zero so we can store sub() result
//             0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
//             0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
//         ));
//         let underflow = sub(result, BASE_MODULUS, &dummy);
//         // underflow==1 means “we tried to go negative,” but that shouldn’t happen
//         if (underflow == 0u) {
//             // `dummy` now holds the reduced result
//             result = dummy;
//         }
//         // else if underflow==1, it means the original result < MOD, so do nothing
//     }

//     return field_reduce(result); 
// }

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
