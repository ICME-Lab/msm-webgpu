use ff::PrimeField;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, FromPrimitive};
use num_integer::Integer;
use crate::{cuzk::msm::calc_num_words, halo2curves::utils::field_to_bytes, utils::montgomery::{bytes_to_field_montgomery, field_to_bytes_montgomery}};

  
//   /**
//  * Converts a single bigint to a Uint8Array for GPU processing, breaking it down into words.
//  * @param {bigint} val - The bigint to convert.
//  * @param {number} num_words - The number of words per bigint.
//  * @param {number} word_size - The size of each word in bits.
//  * @returns {Uint8Array} The resulting byte array for GPU use.
//  */
// export const bigint_to_u8_for_gpu = (
//     val: bigint,
//     num_words: number,
//     word_size: number,
//   ): Uint8Array => {
//     const result = new Uint8Array(num_words * 4);
//     const limbs = to_words_le(BigInt(val), num_words, word_size);
//     for (let i = 0; i < limbs.length; i++) {
//       const i4 = i * 4;
//       result[i4] = limbs[i] & 255;
//       result[i4 + 1] = limbs[i] >> 8;
//     }
  
//     return result;
//   };

pub fn field_to_u8_vec_montgomery_for_gpu<F: PrimeField>(
    field: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    let bytes = field_to_bytes_montgomery(field);

    // TODO: Avoid this step
    let v = BigInt::from_bytes_le(Sign::Plus, &bytes);

    let limbs = to_words_le(&v, num_words, word_size);
    let mut u8_vec = vec![0u8; num_words * 4];

    for (i, limb) in limbs.iter().enumerate() {
        let i4 = i * 4;
        u8_vec[i4] = (limb & 255) as u8;
        u8_vec[i4 + 1] = (limb >> 8) as u8;
    }

    u8_vec
}

pub fn field_to_u8_vec_for_gpu<F: PrimeField>(
    field: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    let bytes = field_to_bytes(field);

    // TODO: Avoid this step
    let v = BigInt::from_bytes_le(Sign::Plus, &bytes);

    let limbs = to_words_le(&v, num_words, word_size);
    let mut u8_vec = vec![0u8; num_words * 4];

    for (i, limb) in limbs.iter().enumerate() {
        let i4 = i * 4;
        u8_vec[i4] = (limb & 255) as u8;
        u8_vec[i4 + 1] = (limb >> 8) as u8;
    }

    u8_vec
}

// TODO: Test
pub fn to_words_le(
    val: &BigInt,
    num_words: usize,
    word_size: usize,
) -> Vec<u16> {
    let mut limbs = vec![0u16; num_words];

    let mask = BigInt::from((1u32 << word_size) - 1);
    for i in 0..num_words {
        let idx = num_words - 1 - i;
        let shift = idx * word_size;
        let w = (val >> shift) & mask.clone();
        let digits = w.to_u32_digits().1;
        if digits.len() > 0 {
            limbs[idx] = digits[0] as u16;
        }
    }   

    limbs
}

pub fn u8s_to_fields_without_assertion<F: PrimeField>(
    u8s: &[u8],
    num_words: usize,
    word_size: usize,
) -> Vec<F> {
    let num_u8s_per_scalar = num_words * 4;

    let mut result = vec![];
    for i in 0..(u8s.len() / num_u8s_per_scalar) {
        let p = i * num_u8s_per_scalar;
        let s = u8s[p..p + num_u8s_per_scalar].to_vec();
        result.push(u8s_to_field_without_assertion(&s, num_words, word_size));
    }
    result
}

pub fn u8s_to_field_without_assertion<F: PrimeField>(
    u8s: &[u8],
    num_words: usize,
    word_size: usize,
) -> F {
    let a = bytemuck::cast_slice::<u8, u16>(u8s);
    let mut limbs = vec![];
    for i in (0..a.len()).step_by(2) {
        limbs.push(a[i]);
    }
    println!("a: {:?}", a);
    println!("limbs: {:?}", limbs);

    from_words_le_without_assertion(&limbs, num_words, word_size)
}

pub fn from_words_le_without_assertion<F: PrimeField>(
    limbs: &[u16],
    num_words: usize,
    word_size: usize,
) -> F {
    let mut val = BigInt::ZERO;
    for i in 0..num_words {
        let exponent = (num_words - i - 1) * word_size;
        val += BigInt::from(2).pow(exponent as u32) * BigInt::from(limbs[num_words - i - 1]);
    }
    let bytes = val.to_bytes_le().1;
    let field = bytes_to_field_montgomery(&bytes);
    field
}

pub fn gen_p_limbs(
    p: &BigInt,
    num_words: usize,
    word_size: usize,
) -> String {
    let limbs = to_words_le(p, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    p.limbs[{}u] = {}u;\n", i, limb);
    }
    r
}

pub fn gen_p_limbs_plus_one(
    p: &BigInt,
    num_words: usize,
    word_size: usize,
) -> String {
    let limbs = to_words_le(p, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    p.limbs[{}u] = {}u;\n", i, limb);
    }
    r += &format!("    p.limbs[{}u] = {}u;\n", limbs.len(), 0);
    r
}

pub fn gen_zero_limbs(
    num_words: usize,
) -> String {
    let mut r = String::new();
    for _i in 0..(num_words - 1) {
        r += &format!("0u, ");
    }
    r += &format!("0u");
    r
}

pub fn gen_r_limbs(
    r: &BigInt,
    num_words: usize,
    word_size: usize,
) -> String {
    let limbs = to_words_le(r, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    r.limbs[{}u] = {}u;\n", i, limb);
    }
    r
}

#[derive(Debug)]
pub struct MiscParams {
    pub num_words: usize,
    pub n0: u32,
    pub r: BigInt,
}

pub fn compute_misc_params(
    p: &BigInt,
    word_size: usize,
) -> MiscParams {
    assert!(word_size > 0);

    let num_words = calc_num_words(word_size);
    let r = BigInt::one() << (num_words * word_size);
    let gcd = r.extended_gcd(p);
    let rinv = gcd.x;
    let pprime = gcd.y;

    if rinv < BigInt::ZERO {
        assert_eq!(((r.clone() * rinv.clone() - p.clone() * pprime.clone()) % p) + p, BigInt::one());
        assert_eq!(((r.clone() * rinv.clone()) % p) + p, BigInt::one());
        assert_eq!((p.clone() * pprime.clone()) % r.clone(), BigInt::one());
      } else {
        assert_eq!((r.clone() * rinv.clone() - p.clone() * pprime.clone()) % p, BigInt::one());
        assert_eq!((r.clone() * rinv.clone()) % p, BigInt::one());
        assert_eq!(((p.clone() * pprime.clone()) % r.clone()) + r.clone(), BigInt::one());
      }
    let neg_n_inv = r.clone() - pprime.clone();
    let n0 = neg_n_inv % (BigInt::one() << word_size);
    let n0_u32 = n0.to_u32_digits();
    assert!(n0_u32.1.len() == 1);
    assert!(n0_u32.0 == Sign::Plus);
    MiscParams { num_words, n0: n0_u32.1[0], r }
}