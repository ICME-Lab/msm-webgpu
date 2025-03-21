use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::identities::Zero;

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

pub fn u16s_to_bigint_cast(b: Vec<u32>) -> Vec<BigUint> {
    assert!(b.len() % 16 == 0);
    b
        .into_iter()
        .chunks(16)
        .into_iter()
        .map(|c| {
            let v : Vec<u32> = c.into_iter().collect();
            limbs_to_bigint256(&v)
        })
        .collect()
}

pub fn u16s_to_bigint(digits: &[u16]) -> BigUint {
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

pub fn bigint_to_u16s(a: &BigUint) -> Vec<u16> {
    let bytes = a.to_bytes_le().to_vec();
    bytemuck::cast_slice::<u8, u16>(&bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use num_traits::Num;

    use super::*;

    #[test]
    fn test_roundtrip() {
        let fp = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941560715954676764349967630337", 10).unwrap();
        let r = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941647379679742748393362948097", 10).unwrap();
        let base_m = BigUint::from_str_radix("115792089237316195423570985008687907853087743403514885215096460958426388758524", 10).unwrap();

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);
        let base_m_bytes = bigint_to_u16s(&base_m);

        assert_eq!(fp, u16s_to_bigint(&fp_bytes));
        assert_eq!(r, u16s_to_bigint(&r_bytes));
        assert_eq!(base_m, u16s_to_bigint(&base_m_bytes)); 
    }

    #[test]
    fn test_wgsl_pallas_constants() {
        let fp = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941560715954676764349967630337", 10).unwrap();
        let r = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941647379679742748393362948097", 10).unwrap();
        let base_m = BigUint::from_str_radix("115792089237316195423570985008687907853087743403514885215096460958426388758524", 10).unwrap();
        let u = BigUint::from_str_radix("45560315531419706090280762371685220353", 10).unwrap();

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);
        let base_m_bytes = bigint_to_u16s(&base_m);
        let u_bytes = bigint_to_u16s(&u);

        println!("u_bytes: {:?}", u_bytes);

        // These are the values hardcoded in pallas.wgsl
        assert_eq!(fp_bytes, vec![1, 0, 12525, 39213, 63771, 2380, 39164, 8774, 0, 0, 0, 0, 0, 0, 0, 16384]);
        assert_eq!(r_bytes, vec![1, 0, 60193, 35910, 43229, 2452, 39164, 8774, 0, 0, 0, 0, 0, 0, 0, 16384]);
        assert_eq!(base_m_bytes, vec![65532, 65535, 15435, 39755, 7057, 56012, 39951, 30437, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535]);
    }

    #[test]
    fn test_wgsl_bn254_constants() {
        let fp = BigUint::from_str_radix("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10).unwrap();
        let r = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        let base_m = BigUint::from_str_radix("7414231717174750794300032619171286607037563074092753157756839893656085003591", 10).unwrap();

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);
        let base_m_bytes = bigint_to_u16s(&base_m);

        // These are the values hardcoded in pallas.wgsl
        assert_eq!(fp_bytes, vec![64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
        assert_eq!(r_bytes, vec![1, 61440, 62867, 17377, 28817, 31161, 59464, 10291, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
        assert_eq!(base_m_bytes, vec![64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 4196]);
    }
}
