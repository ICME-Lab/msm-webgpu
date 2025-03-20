use ark_pallas::Fr;
use ark_ff::{BigInt, PrimeField, Field};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::identities::Zero;

pub fn fr_vec_to_biguint_vec(vals: &Vec<Fr>) -> Vec<BigUint> {
    vals.iter().map(|v| (*v).into()).collect()
}

pub fn concat_files(filenames: Vec<&str>) -> String {
    let mut result = String::new();

    for (i, filename) in filenames.iter().enumerate() {
        let contents = std::fs::read_to_string(filename).unwrap();
        result += &String::from(format!("//---- {}\n\n", filename));
        result += &String::from(contents);
        if i < filenames.len() - 1 {
            result += "\n";
        }
    }
    String::from(result)
}

/// Converts an array of 16 limbs to a BigUint.
pub fn limbs_to_bigint256(limbs: &[u32]) -> BigUint {
    assert!(limbs.len() == 16);
    let mut res = BigUint::zero();
    for (i, limb) in limbs.iter().enumerate() {
        res += BigUint::from_slice(&[2]).pow((i * 16).try_into().unwrap())
            * BigUint::from_slice(&[limb.clone()]);
    }

    res
}

pub fn split_biguint(a: BigUint) -> Vec<u8> {
    // Convert the input to bytes
    let mut a_bytes = a.to_bytes_le().to_vec();
    assert!(a_bytes.len() <= 32);

    // Pad the byte vector with 0s such that the final length is 32
    while a_bytes.len() < 32 {
        a_bytes.push(0u8);
    }

    let mut result = Vec::with_capacity(64);
    let mut i = 0;
    loop {
        if i >= a_bytes.len() {
            break;
        }

        result.push(a_bytes[i]);
        result.push(a_bytes[i + 1]);
        result.push(0u8);
        result.push(0u8);
        i += 2;
    }

    result
}

pub fn fields_to_u16_vec<F: PrimeField>(fields: &[F]) -> Vec<u16> {
    fields
        .iter()
        .map(|field| field_to_u16_vec(field))
        .flatten()
        .collect()
}

pub fn field_to_u16_vec<F: PrimeField>(field: &F) -> Vec<u16> {
    let mut bytes = Vec::new();
    field.serialize_uncompressed(&mut bytes).unwrap();
    let mut u16_vec = Vec::new();

    for chunk in bytes.chunks_exact(2) {
        u16_vec.push(u16::from_le_bytes(chunk.try_into().unwrap()));
    }

    // u16_vec.reverse();
    u16_vec
}

pub fn bigints_to_bytes(vals: Vec<BigUint>) -> Vec<u8> {
    let mut input_as_bytes: Vec<Vec<u8>> = Vec::with_capacity(vals.len());
    for i in 0..vals.len() {
        input_as_bytes.push(split_biguint(vals[i].clone()));
    }

    input_as_bytes.into_iter().flatten().collect()
}

pub fn point_to_bytes(vals: &Vec<(BigUint, BigUint, BigUint)>) -> Vec<u8> {
    let mut input_as_bytes: Vec<Vec<u8>> = Vec::with_capacity(vals.len() * 3);

    for (x, y, z) in vals {
        input_as_bytes.push(split_biguint(x.clone()));
        input_as_bytes.push(split_biguint(y.clone()));
        input_as_bytes.push(split_biguint(z.clone()));
    }

    input_as_bytes.into_iter().flatten().collect()
}

pub fn u32s_to_bigints(b: Vec<u32>) -> Vec<BigUint> {
    assert!(b.len() % 16 == 0);
    let chunks: Vec<Vec<u32>> = b
        .into_iter()
        .chunks(16)
        .into_iter()
        .map(|c| c.into_iter().collect())
        .collect();

    chunks.iter().map(|c| limbs_to_bigint256(c)).collect()
}

pub fn u16_array_to_fields<F: PrimeField>(u16_array: &[u16]) -> Vec<F> {
    u16_array
        .chunks_exact(16) // Each field element is 16 u16s (16 * 16 bits)
        .map(|chunk| {
            // reversed_chunk.reverse();
            let bytes = chunk.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>();
            F::from_le_bytes_mod_order(&bytes)

            // let biguint = u16_array_to_big_int(&reversed_chunk);
            // bigint_to_field(biguint)
        })
        .collect()
}


pub fn u16_array_to_big_int(digits: &[u16]) -> BigUint {
    // Set the base to 2^16
    let base = BigUint::from(1u32 << 16);
    let mut result = BigUint::from(0u32);
    let mut multiplier = BigUint::from(1u32);

    // For each digit, add digit * (base^position)
    for &digit in digits {
        result += &multiplier * BigUint::from(digit);
        multiplier *= &base;
    }
    result
}