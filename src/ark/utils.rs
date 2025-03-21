use ark_ff::PrimeField;


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

pub fn u16_vec_to_fields<F: PrimeField>(u16_array: &[u16]) -> Vec<F> {
    u16_array
        .chunks_exact(16) // Each field element is 16 u16s (16 * 16 bits)
        .map(|chunk| {
            // reversed_chunk.reverse();
            let bytes = chunk
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>();
            F::from_le_bytes_mod_order(&bytes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_pallas::Fr;
    #[test]
    fn test_fields_to_u16_vec() {
        let fields: Vec<Fr> = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let u16_vec = fields_to_u16_vec(&fields);
        println!("u16_vec: {:?}", u16_vec);
        let new_fields = u16_vec_to_fields(&u16_vec);
        assert_eq!(fields, new_fields);
    }
}
