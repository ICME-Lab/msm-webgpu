struct BigInt {
    limbs: array<u32, {{ num_words }}>
}

struct BigIntWide {
    limbs: array<u32, {{ num_words_mul_two }}>
}

struct BigIntMediumWide {
    limbs: array<u32, {{ num_words_plus_one }}>
}

struct Point {
  x: BigInt,
  y: BigInt,
  z: BigInt
}

fn bigint_equal(a: BigInt, b: BigInt) -> bool {
    for (var i = 0u; i < {{ num_words }}; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) {
            return false;
        }
    }
    return true;
}