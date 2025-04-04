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

pub fn load_shader_code_pallas() -> String {
    let mut all = String::new();
    // all.push_str(include_str!("./wgsl/pallas/_all.wgsl"));
    all.push_str(include_str!("./wgsl/bigint.wgsl"));
    all.push_str(include_str!("./wgsl/pallas/field.wgsl"));
    all.push_str(include_str!("./wgsl/pallas/curve.wgsl"));
    all.push_str(include_str!("./wgsl/storage.wgsl"));
    all.push_str(include_str!("./wgsl/pippenger.wgsl"));
    all.push_str(include_str!("./wgsl/main.wgsl"));
    all
}

pub fn load_shader_code_bn254() -> String {
    let mut all = String::new();
    // all.push_str(include_str!("./wgsl/bn254/_all.wgsl"));
    all.push_str(include_str!("./wgsl/bigint.wgsl"));
    all.push_str(include_str!("./wgsl/bn254/field.wgsl"));
    all.push_str(include_str!("./wgsl/bn254/curve.wgsl"));
    all.push_str(include_str!("./wgsl/storage.wgsl"));
    all.push_str(include_str!("./wgsl/pippenger.wgsl"));
    all.push_str(include_str!("./wgsl/main.wgsl"));
    all
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
    let mut v = bytemuck::cast_slice::<u8, u16>(&bytes).to_vec();

    while v.len() < 16 {
        v.push(0u16);
    }

    v
}

#[cfg(test)]
mod tests {
    
    use num_integer::Integer;
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
        let two_pow_256 = BigUint::from_str_radix("115792089237316195423570985008687907853269984665640564039457584007913129639936", 10).unwrap();
        let two_pow_254 = BigUint::from_str_radix("28948022309329048855892746252171976963317496166410141009864396001978282409984", 10).unwrap();
        let fp = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941560715954676764349967630337", 10).unwrap();
        let r = BigUint::from_str_radix("28948022309329048855892746252171976963363056481941647379679742748393362948097", 10).unwrap();
        let base_m = BigUint::from_str_radix("115792089237316195423570985008687907853087743403514885215096460958426388758524", 10).unwrap();
        // p = 2^254 + u
        let u = BigUint::from_str_radix("45560315531419706090280762371685220353", 10).unwrap();
        assert_eq!(fp.clone(), two_pow_254.clone() + u.clone());

        // 4 * (2^254 - u) `mod` 2^256
        let t =  BigUint::from(4u32) * (two_pow_254.clone() - u.clone());
        assert_eq!(t.mod_floor(&two_pow_256), base_m);

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);
        let base_m_bytes = bigint_to_u16s(&base_m);
        let u_bytes = bigint_to_u16s(&u);

        // These are the values hardcoded in pallas.wgsl
        assert_eq!(fp_bytes, vec![1, 0, 12525, 39213, 63771, 2380, 39164, 8774, 0, 0, 0, 0, 0, 0, 0, 16384]);
        assert_eq!(r_bytes, vec![1, 0, 60193, 35910, 43229, 2452, 39164, 8774, 0, 0, 0, 0, 0, 0, 0, 16384]);
        assert_eq!(base_m_bytes, vec![65532, 65535, 15435, 39755, 7057, 56012, 39951, 30437, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535]);
        assert_eq!(u_bytes, vec![1, 0, 12525, 39213, 63771, 2380, 39164, 8774, 0, 0, 0, 0, 0, 0, 0, 0]);

        use group::Group;
        use halo2curves::{pasta::pallas::{Point, Base}, CurveExt};
        assert_eq!(Point::identity(), Point::new_jacobian(Base::zero(), Base::zero(), Base::zero()).unwrap());
    }

    #[test]
    fn test_wgsl_bn254_constants() {
        let two_pow_253 = BigUint::from_str_radix("14474011154664524427946373126085988481658748083205070504932198000989141204992", 10).unwrap();
        let two_pow_256 = BigUint::from_str_radix("115792089237316195423570985008687907853269984665640564039457584007913129639936", 10).unwrap();
        let fp = BigUint::from_str_radix("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10).unwrap();
        let r = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        let base_m = BigUint::from_str_radix("28239117749959094534585362027658807498484740036449269388701432429332224805604", 10).unwrap();
        let u = BigUint::from_str_radix("7414231717174750794300032619171286607037563074092753157756839893656085003591", 10).unwrap();

        // 4 * (2^253 - u) `mod` 2^256
        let t =  BigUint::from(4u32) * (two_pow_253.clone() - u.clone());
        assert_eq!(t.mod_floor(&two_pow_256), base_m);

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);
        let base_m_bytes = bigint_to_u16s(&base_m);
        let u_bytes = bigint_to_u16s(&u);

        // These are the values hardcoded in bn254.wgsl
        assert_eq!(fp_bytes, vec![64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
        assert_eq!(r_bytes, vec![1, 61440, 62867, 17377, 28817, 31161, 59464, 10291, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
        assert_eq!(base_m_bytes, vec![2788, 40460, 53156, 3965, 54731, 24120, 21946, 41466, 40585, 63994, 59685, 7870, 32601, 31545, 50740, 15982]);
        assert_eq!(u_bytes, vec![64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 4196]);

        use group::Group;
        use halo2curves::{bn256::{G1, Fq}, CurveExt};
        assert_eq!(G1::identity(), G1::new_jacobian(Fq::zero(), Fq::one(), Fq::zero()).unwrap());
    }
}
