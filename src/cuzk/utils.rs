use num_bigint::BigUint;

// /**
//  * Converts a bigint into an array of 16-bit words in little-endian format.
//  * @param {bigint} val - The bigint to convert.
//  * @param {number} num_words - The number of words.
//  * @param {number} word_size - The size of each word in bits.
//  * @returns {Uint16Array} The array of 16-bit words.
//  */
// export const to_words_le = (
//     val: bigint,
//     num_words: number,
//     word_size: number,
//   ): Uint16Array => {
//     const words = new Uint16Array(num_words);
  
//     const mask = BigInt(2 ** word_size - 1);
//     for (let i = 0; i < num_words; i++) {
//       const idx = num_words - 1 - i;
//       const shift = BigInt(idx * word_size);
//       const w = (val >> shift) & mask;
//       words[idx] = Number(w);
//     }
  
//     return words;
//   };
  
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