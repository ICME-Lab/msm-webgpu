const W = 16u;
const W_mask = (1 << W) - 1u;
const L = 256u;
const N = 16u;
const NN = N * 2u;
const N1 = N + 1u;

// No overflow
struct BigInt256 {
    limbs: array<u32,N> // each limb is a 16-bit value in the low half of the u32
}

struct BigInt512 {
    limbs: array<u32,NN>
}

struct BigInt272 {
    limbs: array<u32,N1>
}

// Careful, a and res may point to the same thing.
fn add(a: BigInt256, b: BigInt256, res: ptr<function, BigInt256>) -> u32 {
    var carry: u32 = 0;
    for (var i: u32 = 0; i < N; i = i + 1u) {
        let c = a.limbs[i] + b.limbs[i] + carry;
        (*res).limbs[i] = c & W_mask;
        carry = c >> W;
    }
    return carry;
}
 
// assumes a >= b
fn sub(a: BigInt256, b: BigInt256, res: ptr<function, BigInt256>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < N; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

// repeated code pls fix
fn add_512(a: BigInt512, b: BigInt512, res: ptr<function, BigInt512>) -> u32 {
    var carry: u32 = 0;
    for (var i: u32 = 0; i < (2*N); i = i + 1u) {
        let c = a.limbs[i] + b.limbs[i] + carry;
        (*res).limbs[i] = c & W_mask;
        carry = c >> W;
    }
    return carry;
}
 
// assumes a >= b
fn sub_512(a: BigInt512, b: BigInt512, res: ptr<function, BigInt512>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < (2*N); i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

// assumes a >= b
fn sub_272(a: BigInt272, b: BigInt272, res: ptr<function, BigInt272>) -> u32 {
    var borrow: u32 = 0;
    for (var i: u32 = 0; i < N + 1; i = i + 1u) {
        (*res).limbs[i] = a.limbs[i] - b.limbs[i] - borrow;
        if (a.limbs[i] < (b.limbs[i] + borrow)) {
            (*res).limbs[i] += W_mask + 1;
            borrow = 1u;
        } else {
            borrow = 0u;
        }
    }
    return borrow;
}

fn mul(a: BigInt256, b: BigInt256) -> BigInt512 {
    var res: BigInt512;
    for (var i = 0u; i < N; i = i + 1u) {
        for (var j = 0u; j < N; j = j + 1u) {
            let c = a.limbs[i] * b.limbs[j];
            res.limbs[i+j] += c & W_mask;
            res.limbs[i+j+1] += c >> W;
        }   
    }
    // start from 0 and carry the extra over to the next index
    for (var i = 0u; i < 2*N - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> W;
        res.limbs[i] = res.limbs[i] & W_mask;
    }
    return res;
}

fn sqr(a: BigInt256) -> BigInt512 {
    var res: BigInt512;
    for (var i = 0u;i < N; i = i + 1u) {
        let sc = a.limbs[i] * a.limbs[i];
        res.limbs[(i << 1)] += sc & W_mask;
        res.limbs[(i << 1)+1] += sc >> W;

        for (var j = i + 1;j < N;j = j + 1u) {
            let c = a.limbs[i] * a.limbs[j];
            res.limbs[i+j] += (c & W_mask) << 1;
            res.limbs[i+j+1] += (c >> W) << 1;
        }
    }

    for (var i = 0u; i < 2*N - 1; i = i + 1u) {
        res.limbs[i+1] += res.limbs[i] >> W;
        res.limbs[i] = res.limbs[i] & W_mask;
    }
    return res;
}

fn to_bytes(s: BigInt256) -> array<u32, 2 * N> {
    // Declare a local array of 32 elements
    var res: array<u32, 2 * N>;

    for (var i = 0u; i < 2 * N; i = i + 1u) {
       res[i] = 0u;
    }

    // For each 16-bit limb, split into two 8-bit bytes
    for (var i: u32 = 0u; i < N; i = i + 1u) {
        let limb: u32 = s.limbs[i];

        // Low 8 bits
        let low_byte:  u32 = limb & 0xFFu;

        // Next 8 bits
        let next_byte: u32 = (limb >> 8u) & 0xFFu;

        // Store the two bytes in consecutive array slots
        res[2u * i] = low_byte;
        res[2u * i + 1u] = next_byte;
    }

    return res;
}