use ff::{PrimeField, Field};
use halo2curves::CurveAffine;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, FromPrimitive};
use num_integer::Integer;
use web_sys::console;
use crate::{cuzk::msm::{calc_num_words, P}, halo2curves::utils::{bytes_to_field, field_to_bytes}, utils::montgomery::{bytes_to_field_montgomery, field_to_bytes_montgomery}};

  
pub fn field_to_u8_vec_montgomery_for_gpu<F: PrimeField>(
    field: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    let bytes = field_to_bytes_montgomery(field);
    let limbs = to_words_le_from_le_bytes(&bytes, num_words, word_size);
    let mut u8_vec = vec![0u8; num_words * 4];

    for (i, limb) in limbs.iter().enumerate() {
        let i4 = i * 4;
        u8_vec[i4] = (limb & 255) as u8;
        u8_vec[i4 + 1] = (limb >> 8) as u8;
    }

    u8_vec
}

pub fn points_to_bytes_for_gpu<C: CurveAffine>(
    g: &Vec<C>,
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    g.into_iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_u8_vec_montgomery_for_gpu(coords.x(), num_words, word_size);
            let y = field_to_u8_vec_montgomery_for_gpu(coords.y(), num_words, word_size);
            let z = field_to_u8_vec_montgomery_for_gpu(&C::Base::ONE, num_words, word_size);
            [x, y, z].concat()
        })
        .collect::<Vec<_>>()
}

pub fn points_to_bytes_for_gpu_x_y<C: CurveAffine>(
    g: &[C],            
    num_words: usize,
    word_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    // each coordinate is num_words * word_size bytes long
    let bytes_per_coord = num_words * 4;

    // pre‑allocate the exact size we need to avoid reallocations
    let mut xs = Vec::with_capacity(g.len() * bytes_per_coord);
    let mut ys = Vec::with_capacity(g.len() * bytes_per_coord);

    for affine in g {
        let coords = affine.coordinates().unwrap();

        xs.extend(field_to_u8_vec_montgomery_for_gpu(
            coords.x(),
            num_words,
            word_size,
        ));
        ys.extend(field_to_u8_vec_montgomery_for_gpu(
            coords.y(),
            num_words,
            word_size,
        ));
    }

    (xs, ys)
}

pub fn fields_to_u8_vec_for_gpu<F: PrimeField>(
    fields: &[F],
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    fields.iter().flat_map(|field| field_to_u8_vec_for_gpu(field, num_words, word_size)).collect::<Vec<_>>()
}

pub fn field_to_u8_vec_for_gpu<F: PrimeField>(
    field: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    let bytes = field_to_bytes(field);

    let limbs = to_words_le_from_le_bytes(&bytes, num_words, word_size);
    let mut u8_vec = vec![0u8; num_words * 4];

    for (i, limb) in limbs.iter().enumerate() {
        let i4 = i * 4;
        u8_vec[i4] = (limb & 255) as u8;
        u8_vec[i4 + 1] = (limb >> 8) as u8;
    }

    u8_vec
}

pub fn to_words_le(
    val: &BigInt,
    num_words: usize,
    word_size: usize,
) -> Vec<u32> {
    let mut limbs = vec![0u32; num_words];

    let mask = BigInt::from((1u32 << word_size) - 1);
    for i in 0..num_words {
        let idx = num_words - 1 - i;
        let shift = idx * word_size;
        let w = (val >> shift) & mask.clone();
        let digits = w.to_u32_digits().1;
        if digits.len() > 0 {
            limbs[idx] = digits[0] as u32;
        }
    }   

    limbs
}

pub fn to_words_le_from_field<F: PrimeField>(
    val: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u32> {
    let bytes = field_to_bytes(val);
    to_words_le_from_le_bytes(&bytes, num_words, word_size)
}

pub fn to_words_le_from_field_montgomery<F: PrimeField>(
    val: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u32> {
    let bytes = field_to_bytes_montgomery(val);
    to_words_le_from_le_bytes(&bytes, num_words, word_size)
}

pub fn to_words_le_from_le_bytes(
    val: &[u8],
    num_words: usize,
    word_size: usize,
) -> Vec<u32> {
    assert!(word_size <= 32, "u32 supports up to 32 bits");

    let mut limbs = vec![0u32; num_words];

    for idx in 0..num_words {
        let mut word = 0u32;

        // Pick out `word_size` bits that start at bit `idx * word_size`
        for bit_in_word in 0..word_size {
            let global_bit = idx * word_size + bit_in_word;
            let byte_idx   = global_bit / 8;          // 0 = least-significant byte
            if byte_idx >= val.len() { break; }       // past the supplied data → 0

            let bit_in_byte = global_bit % 8;
            let bit = (val[byte_idx] >> bit_in_byte) & 1;
            word |= (bit as u32) << bit_in_word;
        }

        limbs[idx] = word;
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
    from_words_le_without_assertion(&limbs, num_words, word_size)
}

pub fn from_words_le_without_assertion<F: PrimeField>(
    limbs: &[u16],
    num_words: usize,
    word_size: usize,
) -> F {
    assert!(num_words == limbs.len());

    let mut val = BigInt::ZERO;
    for i in 0..num_words {
        let exponent = (num_words - i - 1) * word_size;
        let limb = limbs[num_words - i - 1];
        val += BigInt::from(2).pow(exponent as u32) * BigInt::from(limb);
        if val == *P {
            val = BigInt::ZERO;
        }
    }
    // debug(&format!("val: {:?}", val));
    let bytes = val.to_bytes_le().1;
    // debug(&format!("bytes: {:?}", bytes));
    let field = bytes_to_field_montgomery(&bytes);
    // let field = bytes_to_field(&bytes);
    // debug(&format!("field: {:?}", field));
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
    MiscParams { num_words, n0: n0_u32.1[0], r: r % p }
}

pub fn debug(s: &str) {
    // if wasm
    #[cfg(target_arch = "wasm32")]
    console::log_1(&s.into());
    // if not wasm
    #[cfg(not(target_arch = "wasm32"))]
    println!("{}", s);
}

#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fq, Fr};
    use num_traits::Num;
    use rand::{thread_rng, Rng};

    use crate::cuzk::{lib::sample_scalars, msm::{PARAMS, WORD_SIZE}};

    use super::*;
    
    #[test]
    fn test_to_words_le_from_le_bytes() {
        let val = sample_scalars::<Fr>(1)[0];
        let bytes = field_to_bytes(&val);
        for word_size in 13..17 {
            let num_words = calc_num_words(word_size);

            let v = BigInt::from_bytes_le(Sign::Plus, &bytes);
            let limbs = to_words_le(&v, num_words, word_size);
            let limbs_from_le_bytes = to_words_le_from_le_bytes(&bytes, num_words, word_size);
            assert_eq!(limbs, limbs_from_le_bytes);
        }
    }

    #[test]
    fn test_gen_p_limbs() {
        let p = P.clone();
        let num_words = calc_num_words(13);
        let p_limbs = gen_p_limbs(&p, num_words, 13);
        println!("{}", p_limbs);
    }

    #[test]
    fn test_gen_r_limbs() {
        let r = PARAMS.r.clone();
        let num_words = calc_num_words(WORD_SIZE);
        let r_limbs = gen_r_limbs(&r, num_words, WORD_SIZE);
        println!("{}", r_limbs);
    }

    #[test]
    fn test_field_to_u8_vec_montgomery_for_gpu() {
        // random
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        for word_size in 13..17 {
            let num_words = calc_num_words(word_size);
            let bytes = field_to_u8_vec_montgomery_for_gpu(&a, num_words, word_size);
            let a_from_bytes = u8s_to_field_without_assertion(&bytes, num_words, word_size);
            assert_eq!(a, a_from_bytes);
        }
    }


}
