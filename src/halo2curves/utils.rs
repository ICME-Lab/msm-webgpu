use ff::PrimeField;
use halo2curves::bn256::MontgomeryRepr;
use num_bigint::BigUint;


pub fn field_to_bytes<F: PrimeField>(value: F) -> Vec<u8> {
    let s_bytes = value.to_repr();
    let s_bytes_ref = s_bytes.as_ref();
    s_bytes_ref.to_vec()
}

pub fn field_to_bytes_montgomery<F: MontgomeryRepr>(value: F) -> Vec<u8> {
    let s_bytes = value.to_montgomery_repr();
    let s_bytes_ref = s_bytes.as_ref();
    s_bytes_ref.to_vec()
}
pub fn bytes_to_field<F: PrimeField>(bytes: &[u8]) -> F {
    let mut repr = F::Repr::default();
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    F::from_repr(repr).unwrap()
}

pub fn bytes_to_field_montgomery<F: MontgomeryRepr>(bytes: &[u8]) -> F {
    let mut repr = F::Repr::default();
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    F::from_montgomery_repr(repr).unwrap()
}

pub fn fields_to_u16_vec<F: PrimeField>(fields: &[F]) -> Vec<u16> {
    fields
        .iter()
        .map(|field| field_to_u16_vec(field))
        .flatten()
        .collect()
}

pub fn fields_to_u16_vec_montgomery<F: MontgomeryRepr>(fields: &[F]) -> Vec<u16> {
    fields
        .iter()
        .map(|field| field_to_u16_vec_montgomery(field))
        .flatten()
        .collect()
}

pub fn field_to_u16_vec<F: PrimeField>(field: &F) -> Vec<u16> {
    let bytes = field_to_bytes(field.clone());
    let mut u16_vec = Vec::new();

    for chunk in bytes.chunks_exact(2) {
        u16_vec.push(u16::from_le_bytes(chunk.try_into().unwrap()));
    }

    u16_vec
}

pub fn field_to_u16_vec_montgomery<F: MontgomeryRepr>(field: &F) -> Vec<u16> {
    let bytes = field_to_bytes_montgomery(field.clone());
    let mut u16_vec = Vec::new();

    for chunk in bytes.chunks_exact(2) {
        u16_vec.push(u16::from_le_bytes(chunk.try_into().unwrap()));
    }

    u16_vec
}

pub fn u16_vec_to_fields<F: PrimeField>(u16_array: &[u16]) -> Vec<F> {
    u16_array
        .chunks_exact(16) // Each field element is 16 u16s (16 * 16 bits)
        .map(|chunk| {
            let bytes = chunk
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            println!("bytes: {:?}", bytes);
            bytes_to_field::<F>(&bytes)
        })
        .collect()
}

pub fn u16_vec_to_fields_montgomery<F: MontgomeryRepr>(u16_array: &[u16]) -> Vec<F> {
    u16_array
        .chunks_exact(16) // Each field element is 16 u16s (16 * 16 bits)
        .map(|chunk| {
            let bytes = chunk
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            println!("bytes: {:?}", bytes);
            bytes_to_field_montgomery::<F>(&bytes)
        })
        .collect()
}

pub fn cast_u8_to_u16(u8_array: &[u8]) -> Vec<u16> {
    let output_u32: Vec<u32> = bytemuck::cast_slice::<u8, u32>(u8_array).to_vec();
    output_u32
        .iter()
        .map(|&x| {
            if x > u16::MAX as u32 {
                panic!("Value {} is too large for u16", x);
            }
            x as u16
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use halo2curves::{bn256::Fq, pasta::pallas::Scalar as Fr};

    #[test]
    fn test_field_to_bytes() {
        let field = Fr::random(rand::thread_rng());
        let bytes = field_to_bytes(field);
        assert_eq!(bytes.len(), 32);
        let field_from_bytes = bytes_to_field::<Fr>(&bytes);
        assert_eq!(field, field_from_bytes);
    }

    #[test]
    fn test_field_to_bytes_montgomery() {
        let field = Fq::random(rand::thread_rng());
        let bytes = field_to_bytes_montgomery(field);
        assert_eq!(bytes.len(), 32);
        let field_from_bytes = bytes_to_field_montgomery::<Fq>(&bytes);
        assert_eq!(field, field_from_bytes);
    }

    #[test]
    fn test_fields_to_u16_vec() {
        let fields: Vec<Fr> = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let u16_vec = fields_to_u16_vec(&fields);
        println!("u16_vec: {:?}", u16_vec);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_fields_to_u16_vec_montgomery() {
        let fields: Vec<Fq> = vec![Fq::from(1), Fq::from(2), Fq::from(3)];
        let u16_vec = fields_to_u16_vec_montgomery(&fields);
        println!("u16_vec: {:?}", u16_vec);
        let new_fields = u16_vec_to_fields_montgomery(&u16_vec);
        assert_eq!(fields, new_fields);
    }
}
