use num_bigint::BigUint;

  
// TODO: Test
pub fn to_words_le(
    val: &BigUint,
    num_words: usize,
    word_size: usize,
) -> Vec<u16> {
    let mut limbs = vec![0u16; num_words];

    let mask = BigUint::from(1u32 << word_size - 1);
    for i in 0..num_words {
        let idx = num_words - 1 - i;
        let shift = idx * word_size;
        let w = (val >> shift) & mask.clone();
        limbs[idx] = w.to_u32_digits()[0] as u16;
    }   

    limbs
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