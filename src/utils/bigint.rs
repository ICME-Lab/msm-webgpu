use num_bigint::BigUint;

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
        let fp = BigUint::from_str_radix("21888242871839275222246405745257275088696311157297823662689037894645226208583", 10).unwrap();
        let r = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();

        let fp_bytes = bigint_to_u16s(&fp);
        let r_bytes = bigint_to_u16s(&r);


        // These are the values hardcoded in bn254.wgsl
        assert_eq!(fp_bytes, vec![64839, 55420, 35862, 15392, 51853, 26737, 27281, 38785, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
        assert_eq!(r_bytes, vec![1, 61440, 62867, 17377, 28817, 31161, 59464, 10291, 22621, 33153, 17846, 47184, 41001, 57649, 20082, 12388]);
    }
}
