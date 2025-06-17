use ff::{Field, PrimeField};
use halo2curves::CurveAffine;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Num, One};
#[cfg(target_arch = "wasm32")]
use web_sys::console;

pub fn compute_p<C: CurveAffine>() -> BigUint {
    // Trim 0x prefix
    let modulus = C::Base::MODULUS;
    let modulus_str = if modulus.starts_with("0x") {
        &modulus[2..]
    } else {
        modulus
    };

    BigUint::from_str_radix(modulus_str, 16).unwrap()
}

/// Convert a field element to bytes
pub fn field_to_bytes<F: PrimeField>(value: &F) -> Vec<u8> {
    let s_bytes = value.to_repr();
    let s_bytes_ref = s_bytes.as_ref();
    s_bytes_ref.to_vec()
}

/// Convert bytes to a field element
pub fn bytes_to_field<F: PrimeField>(bytes: &[u8]) -> F {
    let mut repr = F::Repr::default();
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    F::from_repr(repr).unwrap()
}

/// Convert a binary representation into u32 limbs.
pub fn to_words_le_from_le_bytes(val: &[u8], num_words: usize, word_size: usize) -> Vec<u32> {
    assert!(word_size <= 32, "u32 supports up to 32 bits");

    let mut limbs = vec![0u32; num_words];

    for (idx, limb) in limbs.iter_mut().enumerate() {
        let mut word = 0u32;

        // Pick out `word_size` bits that start at bit `idx * word_size`
        for bit_in_word in 0..word_size {
            let global_bit = idx * word_size + bit_in_word;
            let byte_idx = global_bit / 8; // 0 = least-significant byte
            if byte_idx >= val.len() {
                break;
            } // past the supplied data → 0

            let bit_in_byte = global_bit % 8;
            let bit = (val[byte_idx] >> bit_in_byte) & 1;
            word |= (bit as u32) << bit_in_word;
        }

        *limb = word;
    }

    limbs
}

/// Convert a vector of u32 limbs into a BigUint
pub fn to_biguint_le(limbs: &[u32], num_limbs: usize, log_limb_size: u32) -> BigUint {
    assert!(limbs.len() == num_limbs);
    let mut res = BigUint::from(0u32);
    let max = 2u32.pow(log_limb_size);

    for i in 0..num_limbs {
        assert!(limbs[i] < max);
        let idx = (num_limbs - 1 - i) as u32;
        let a = idx * log_limb_size;
        let b = BigUint::from(2u32).pow(a) * BigUint::from(limbs[idx as usize]);

        res += b;
    }

    res
}

/// Convert a BigUint into u32 limbs
pub fn to_words_le(val: &BigUint, num_words: usize, word_size: usize) -> Vec<u32> {
    let mut limbs = vec![0u32; num_words];

    let mask = BigUint::from((1u32 << word_size) - 1);
    for i in 0..num_words {
        let idx = num_words - 1 - i;
        let shift = idx * word_size;
        let w = (val >> shift) & mask.clone();
        let digits = w.to_u32_digits();
        if !digits.is_empty() {
            limbs[idx] = digits[0];
        }
    }

    limbs
}

/// Convert a field element into u32 limbs
pub fn to_words_le_from_field<F: PrimeField>(
    val: &F,
    num_words: usize,
    word_size: usize,
) -> Vec<u32> {
    let bytes = field_to_bytes(val);
    to_words_le_from_le_bytes(&bytes, num_words, word_size)
}

/// Split each field element into limbs and convert each limb to a vector of bytes.
pub fn fields_to_u8_vec_for_gpu<F: PrimeField>(
    fields: &[F],
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    fields
        .iter()
        .flat_map(|field| field_to_u8_vec_for_gpu(field, num_words, word_size))
        .collect::<Vec<_>>()
}

/// Split a field element into limbs and convert each limb to a vector of bytes.
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

/// Convert a vector of bytes into a vector of field elements
pub fn u8s_to_fields_without_assertion<F: PrimeField>(
    p: &BigUint,
    u8s: &[u8],
    num_words: usize,
    word_size: usize,
) -> Vec<F> {
    let num_u8s_per_scalar = num_words * 4;

    let mut result = vec![];
    for i in 0..(u8s.len() / num_u8s_per_scalar) {
        let t = i * num_u8s_per_scalar;
        let s = u8s[t..t + num_u8s_per_scalar].to_vec();
        result.push(u8s_to_field_without_assertion(p, &s, num_words, word_size));
    }
    result
}

/// Convert a vector of bytes into a field element
pub fn u8s_to_field_without_assertion<F: PrimeField>(
    p: &BigUint,
    u8s: &[u8],
    num_words: usize,
    word_size: usize,
) -> F {
    let a = bytemuck::cast_slice::<u8, u16>(u8s);
    let mut limbs = vec![];
    for i in (0..a.len()).step_by(2) {
        limbs.push(a[i]);
    }
    from_words_le_without_assertion(p,&limbs, num_words, word_size)
}

/// Convert u16 limbs into a field element
pub fn from_words_le_without_assertion<F: PrimeField>(
    p: &BigUint,
    limbs: &[u16],
    num_words: usize,
    word_size: usize,
) -> F {
    assert!(num_words == limbs.len());

    let mut val = BigUint::ZERO;
    for i in 0..num_words {
        let exponent = (num_words - i - 1) * word_size;
        let limb = limbs[num_words - i - 1];
        val += BigUint::from(2u32).pow(exponent as u32) * BigUint::from(limb);
        if val == *p {
            val = BigUint::ZERO;
        }
    }
    let bytes = val.to_bytes_le();
    
    bytes_to_field(&bytes)
}

/// Convert a vector of points to a vector of bytes
pub fn points_to_bytes_for_gpu<C: CurveAffine>(
    g: &[C],
    num_words: usize,
    word_size: usize,
) -> Vec<u8> {
    g.iter()
        .flat_map(|affine| {
            let coords = affine.coordinates().unwrap();
            let x = field_to_u8_vec_for_gpu(coords.x(), num_words, word_size);
            let y = field_to_u8_vec_for_gpu(coords.y(), num_words, word_size);
            let z = field_to_u8_vec_for_gpu(&C::Base::ONE, num_words, word_size);
            [x, y, z].concat()
        })
        .collect::<Vec<_>>()
}

/// Generate the GPU representation of the field characteristic
pub fn gen_p_limbs(p: &BigUint, num_words: usize, word_size: usize) -> String {
    let limbs = to_words_le(p, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    p.limbs[{i}u] = {limb}u;\n");
    }
    r
}

/// Generate the GPU representation of the field characteristic padded with a zero limb
pub fn gen_p_limbs_plus_one(p: &BigUint, num_words: usize, word_size: usize) -> String {
    let limbs = to_words_le(p, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    p.limbs[{i}u] = {limb}u;\n");
    }
    r += &format!("    p.limbs[{}u] = {}u;\n", limbs.len(), 0);
    r
}

/// Generate the GPU representation of zero
pub fn gen_zero_limbs(num_words: usize) -> String {
    let mut r = String::new();
    for _i in 0..(num_words - 1) {
        r += "0u, ";
    }
    r += "0u";
    r
}

/// Generate the GPU representation of one
pub fn gen_one_limbs(num_words: usize) -> String {
    let mut r = String::new();
    r += "1u, ";
    for _i in 0..(num_words - 2) {
        r += "0u, ";
    }
    r += "0u";
    r
}

/// Generate the GPU representation of the Montgomery radix
pub fn gen_r_limbs(r: &BigUint, num_words: usize, word_size: usize) -> String {
    let limbs = to_words_le(r, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    r.limbs[{i}u] = {limb}u;\n");
    }
    r
}

/// Generate the GPU representation of the Montgomery radix inverse
pub fn gen_rinv_limbs(rinv: &BigUint, num_words: usize, word_size: usize) -> String {
    let limbs = to_words_le(rinv, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    rinv.limbs[{i}u] = {limb}u;\n");
    }
    r
}

/// Generate the Montgomery magic number
pub fn gen_mu(p: &BigUint) -> BigUint {
    let mut x = 1u32;
    let two = BigUint::from(2u32);

    while two.pow(x) < *p {
        x += 1;
    }

    BigUint::from(4u32).pow(x) / p
}

/// Generate the GPU representation of the Montgomery magic number
pub fn gen_mu_limbs(p: &BigUint, num_words: usize, word_size: usize) -> String {
    let mu = gen_mu(p);
    let limbs = to_words_le(&mu, num_words, word_size);
    let mut r = String::new();
    for (i, limb) in limbs.iter().enumerate() {
        r += &format!("    mu.limbs[{i}u] = {limb}u;\n");
    }
    r
}

/// Calculate the bitwidth of the field characteristic
pub fn calc_bitwidth(p: &BigUint) -> usize {
    if *p == BigUint::from(0u32) {
        return 0;
    }

    p.to_radix_le(2).len()
}

/// Extended Euclidean algorithm
fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if *a == BigInt::from(0u32) {
        return (b.clone(), BigInt::from(0u32), BigInt::from(1u32));
    }
    let (g, x, y) = egcd(&(b % a), a);

    (g, y - (b / a) * x.clone(), x.clone())
}

/// Calculate the Montgomery inverse and the Montgomery reduction parameter
pub fn calc_inv_and_pprime(p: &BigUint, r: &BigUint) -> (BigUint, BigUint) {
    assert!(*r != BigUint::from(0u32));

    let p_bigint = BigInt::from_biguint(Sign::Plus, p.clone());
    let r_bigint = BigInt::from_biguint(Sign::Plus, r.clone());
    let one = BigInt::from(1u32);
    let (_, mut rinv, mut pprime) = egcd(
        &BigInt::from_biguint(Sign::Plus, r.clone()),
        &BigInt::from_biguint(Sign::Plus, p.clone()),
    );

    if rinv.sign() == Sign::Minus {
        rinv = BigInt::from_biguint(Sign::Plus, p.clone()) + rinv;
    }

    if pprime.sign() == Sign::Minus {
        pprime = BigInt::from_biguint(Sign::Plus, r.clone()) + pprime;
    }

    // r * rinv - p * pprime == 1
    assert!(
        (BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint)
            - (&p_bigint * &pprime % &p_bigint)
            == one
    );

    // r * rinv % p == 1
    assert!((BigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint) == one);

    // p * pprime % r == 1
    assert!((&p_bigint * &pprime % &r_bigint) == one);

    (rinv.to_biguint().unwrap(), pprime.to_biguint().unwrap())
}

/// Calculate the Montgomery radix inverse and the Montgomery reduction parameter
pub fn calc_rinv_and_n0(p: &BigUint, r: &BigUint, log_limb_size: u32) -> (BigUint, u32) {
    let (rinv, pprime) = calc_inv_and_pprime(p, r);
    let pprime = BigInt::from_biguint(Sign::Plus, pprime);

    let neg_n_inv = BigInt::from_biguint(Sign::Plus, r.clone()) - pprime;
    let n0 = neg_n_inv % BigInt::from(2u32.pow(log_limb_size));
    let n0 = n0.to_biguint().unwrap().to_u32_digits()[0];

    (rinv, n0)
}

/// Miscellaneous parameters for the WebGPU shader
#[derive(Debug, Clone)]
pub struct MiscParams {
    pub num_words: usize,
    pub n0: u32,
    pub r: BigUint,
    pub rinv: BigUint,
    pub p: BigUint,
}

/// Calculate the number of words in the field characteristic
pub fn calc_num_words(p: &BigUint, word_size: usize) -> usize {
    let p_bit_length = calc_bitwidth(p);
    let mut num_words = p_bit_length / word_size;
    while num_words * word_size < p_bit_length {
        num_words += 1;
    }
    num_words
}

/// Compute miscellaneous parameters for the WebGPU shader
pub fn compute_misc_params(p: &BigUint, word_size: usize) -> MiscParams {
    assert!(word_size > 0);
    let num_words = calc_num_words(p, word_size);
    let r = BigUint::one() << (num_words * word_size);
    let res = calc_rinv_and_n0(p, &r, word_size as u32);
    let rinv = res.0;
    let n0 = res.1;
    MiscParams {
        num_words,
        n0,
        r: r % p,
        rinv,
        p: p.clone(),
    }
}

/// Debug print
pub fn debug(s: &str) {
    // if wasm
    #[cfg(target_arch = "wasm32")]
    console::log_1(&s.into());
    // if not wasm
    #[cfg(not(target_arch = "wasm32"))]
    println!("{s}");
}

#[cfg(test)]
mod tests {
    use halo2curves::bn256::{Fq, Fr, Bn256, G1Affine};
    use ff::{Field, PrimeField};
    use num_traits::Num;
    use rand::thread_rng;

    use super::*;
    use crate::cuzk::msm::WORD_SIZE;
    use crate::sample_scalars;

    #[test]
    fn test_to_words_le_from_le_bytes() {
        let p = compute_p::<G1Affine>();
        let val = sample_scalars::<Fr>(1)[0];
        let bytes = field_to_bytes(&val);
        for word_size in 13..17 {
            let num_words = calc_num_words(&p, word_size);

            let v = BigUint::from_bytes_le(&bytes);
            let limbs = to_words_le(&v, num_words, word_size);
            let limbs_from_le_bytes = to_words_le_from_le_bytes(&bytes, num_words, word_size);
            assert_eq!(limbs, limbs_from_le_bytes);
        }
    }

    #[test]
    fn test_gen_p_limbs() {
        let p = compute_p::<G1Affine>();
        let num_words = calc_num_words(&p, WORD_SIZE);
        let p_limbs = gen_p_limbs(&p, num_words, WORD_SIZE);
        println!("{}", p_limbs);
    }

    #[test]
    fn test_gen_r_limbs() {
        let p = compute_p::<G1Affine>();
        let params = compute_misc_params(&p, WORD_SIZE);
        let r = params.r.clone();
        let num_words = calc_num_words(&p, WORD_SIZE);
        let r_limbs = gen_r_limbs(&r, num_words, WORD_SIZE);
        println!("{}", r_limbs);
    }

    #[test]
    fn test_field_to_u8_vec_for_gpu() {
        // random
        let p = compute_p::<G1Affine>();
        let mut rng = thread_rng();
        let a = Fq::random(&mut rng);
        for word_size in 13..17 {
            let num_words = calc_num_words(&p, word_size);
            let bytes = field_to_u8_vec_for_gpu(&a, num_words, word_size);
            let a_from_bytes = u8s_to_field_without_assertion(&p, &bytes, num_words, word_size);
            assert_eq!(a, a_from_bytes);
        }
    }

    #[test]
    fn test_to_words_le() {
        let a = BigUint::from_str_radix(
            "12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
            16,
        )
        .unwrap();
        let limbs = to_words_le(&a, 20, 13);
        let expected = vec![
            1, 0, 0, 768, 4257, 0, 0, 8154, 2678, 2765, 3072, 6255, 4581, 6694, 6530, 5290, 6700,
            2804, 2777, 37,
        ];
        assert_eq!(limbs, expected);
    }
}
