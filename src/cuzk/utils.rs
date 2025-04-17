use ff::PrimeField;
use num_bigint::BigUint;

  
// TODO: Test
pub fn to_words_le(
    val: &BigUint,
    num_words: usize,
    word_size: usize,
) -> Vec<u16> {
    let mut limbs = vec![0u16; num_words];

    let mask = BigUint::from((1u32 << word_size) - 1);
    for i in 0..num_words {
        let idx = num_words - 1 - i;
        let shift = idx * word_size;
        let w = (val >> shift) & mask.clone();
        let digits = w.to_u32_digits();
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

    from_words_le_without_assertion(&limbs, num_words, word_size)
}

pub fn from_words_le_without_assertion<F: PrimeField>(
    limbs: &[u16],
    num_words: usize,
    word_size: usize,
) -> F {
    let mut val = F::ZERO;
    for i in 0..num_words {
        let exponent = (num_words - i - 1) * word_size;
        // TODO: This looks wrong to me. Check Montgomery representation
        val += F::from(2).pow([exponent as u64]) * F::from(limbs[num_words - i - 1] as u64);
    }
    val
}

pub fn gen_p_limbs(
    p: &BigUint,
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
    p: &BigUint,
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