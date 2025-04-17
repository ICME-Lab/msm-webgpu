use ff::PrimeField;

pub fn field_to_bytes_montgomery<F: PrimeField>(value: &F) -> Vec<u8> {
    let s_bytes = value.to_montgomery_repr();
    let s_bytes_ref = s_bytes.as_ref();
    s_bytes_ref.to_vec()
}

pub fn bytes_to_field_montgomery<F: PrimeField>(bytes: &[u8]) -> F {
    let mut repr = F::Repr::default();
    repr.as_mut()[..bytes.len()].copy_from_slice(bytes);
    F::from_montgomery_repr(repr).unwrap()
}

pub fn fields_to_u16_vec_montgomery<F: PrimeField>(fields: &[F]) -> Vec<u16> {
    fields
        .iter()
        .map(|field| field_to_u16_vec_montgomery(field))
        .flatten()
        .collect()
}

pub fn field_to_u16_vec_montgomery<F: PrimeField>(field: &F) -> Vec<u16> {
    let bytes = field_to_bytes_montgomery(field);
    let mut u16_vec = Vec::new();

    for chunk in bytes.chunks_exact(2) {
        u16_vec.push(u16::from_le_bytes(chunk.try_into().unwrap()));
    }

    u16_vec
}

pub fn field_to_u16_as_u32_as_u8_vec_montgomery<F: PrimeField>(field: &F) -> Vec<u8> {
    let bytes = field_to_bytes_montgomery(field);
    assert!(bytes.len() % 2 == 0);
    
    let mut output = Vec::with_capacity((bytes.len() / 2) * 4); // each u16 → u32 → 4 bytes

    for chunk in bytes.chunks_exact(2) {
        let val = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        output.extend_from_slice(&val.to_le_bytes());
    }

    output
}

pub fn u16_vec_to_fields_montgomery<F: PrimeField>(u16_array: &[u16]) -> Vec<F> {
    u16_array
        .chunks_exact(16) // Each field element is 16 u16s (16 * 16 bits)
        .map(|chunk| {
            let bytes = chunk
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            bytes_to_field_montgomery::<F>(&bytes)
        })
        .collect()
}

pub fn fields_to_u32_vec_montgomery<F: PrimeField>(fields: &[F]) -> Vec<u32> {
    fields
        .iter()
        .map(|field| field_to_u32_vec_montgomery(field))
        .flatten()
        .collect()
}

pub fn field_to_u32_vec_montgomery<F: PrimeField>(field: &F) -> Vec<u32> {
    let bytes = field_to_bytes_montgomery(field);
    let mut u32_vec = Vec::new();

    for chunk in bytes.chunks_exact(4) {
        u32_vec.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    u32_vec
}

pub fn u32_vec_to_fields_montgomery<F: PrimeField>(u32_array: &[u32]) -> Vec<F> {
    u32_array
        .chunks_exact(8) // Each field element is 8 u32s (8 * 32 bits)
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


#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use halo2curves::bn256::Fq;

    #[test]
    fn test_field_to_bytes_montgomery() {
        let field = Fq::random(rand::thread_rng());
        let bytes = field_to_bytes_montgomery(&field);
        assert_eq!(bytes.len(), 32);
        let field_from_bytes = bytes_to_field_montgomery::<Fq>(&bytes);
        assert_eq!(field, field_from_bytes);
    }
    #[test]
    fn test_fields_to_u16_vec_montgomery() {
        let fields: Vec<Fq> = vec![Fq::from(1), Fq::from(2), Fq::from(3)];
        let u16_vec = fields_to_u16_vec_montgomery(&fields);
        println!("u16_vec: {:?}", u16_vec);
        let new_fields = u16_vec_to_fields_montgomery(&u16_vec);
        assert_eq!(fields, new_fields);
    }

    #[test]
    fn test_fields_to_u32_vec_montgomery() {
        let fields: Vec<Fq> = vec![Fq::from(1), Fq::from(2), Fq::from(3)];
        let u32_vec = fields_to_u32_vec_montgomery(&fields);
        println!("u32_vec: {:?}", u32_vec);
        let new_fields = u32_vec_to_fields_montgomery(&u32_vec);
        assert_eq!(fields, new_fields);
    }

}