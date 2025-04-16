// This file is curve-dependent:
// The Jacobian ZERO may differ from curve to curve
struct JacobianPoint {
    x: BaseField,
    y: BaseField,
    z: BaseField
};

const JACOBIAN_IDENTITY: JacobianPoint = JacobianPoint(ZERO, ZERO, ZERO);

fn is_inf(p: JacobianPoint) -> bool {
    return field_eq(p.z, ZERO);
}

fn jacobian_double(p: JacobianPoint) -> JacobianPoint {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    let A = montgomery_product(&p.x, &p.x);
    let B = montgomery_product(&p.y, &p.y);
    let C = montgomery_product(&B, &B);
    let X1plusB = field_add(p.x, B);
    let D = field_small_scalar_shift(1, field_sub(field_sqr(X1plusB), field_add(A, C)));
    let E = field_add(field_small_scalar_shift(1, A), A);
    let F = field_sqr(E);
    let x3 = field_sub(F, field_small_scalar_shift(1, D));
    let y3 = field_sub(montgomery_product(E, field_sub(D, x3)), field_small_scalar_shift(3, C));
    let z3 = montgomery_product(field_small_scalar_shift(1, p.y), p.z);
    return JacobianPoint(x3, y3, z3);
}

fn jacobian_add(p: JacobianPoint, q: JacobianPoint) -> JacobianPoint {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
    if (field_eq(p.y, ZERO)) {
        return q;
    }
    if (field_eq(q.y, ZERO)) {
        return p;
    }

    let Z1Z1 = field_sqr(p.z);
    let Z2Z2 = field_sqr(q.z);
    let U1 = montgomery_product(p.x, Z2Z2);
    let U2 = montgomery_product(q.x, Z1Z1);
    let S1 = montgomery_product(p.y, montgomery_product(Z2Z2, q.z));
    let S2 = montgomery_product(q.y, montgomery_product(Z1Z1, p.z));
    if (field_eq(U1, U2)) {
        if (field_eq(S1, S2)) {
            return jacobian_double(p);
        } else {
            return JACOBIAN_IDENTITY;
        }
    }

    let H = field_sub(U2, U1);
    let I = field_small_scalar_shift(2, field_sqr(H));
    let J = montgomery_product(H, I);
    let R = field_small_scalar_shift(1, field_sub(S2, S1));
    let V = montgomery_product(U1, I);
    let nx = field_sub(field_sqr(R), field_add(J, field_small_scalar_shift(1, V)));
    let ny = field_sub(montgomery_product(R, field_sub(V, nx)), field_small_scalar_shift(1, montgomery_product(S1, J)));
    let nz = montgomery_product(H, field_sub(field_pow(field_add(p.z, q.z), 2), field_add(Z1Z1, Z2Z2)));
    return JacobianPoint(nx, ny, nz);
}

fn jacobian_mul(p: JacobianPoint, k: ScalarField) -> JacobianPoint {
    var r: JacobianPoint = JACOBIAN_IDENTITY;
    var t: JacobianPoint = p;
    for (var i = 0u; i < N; i = i + 1u) {
        var k_s = k.limbs[i];
        for (var j = 0u; j < W; j = j + 1u) {
            if ((k_s & 1) == 1u) {
                r = jacobian_add(r, t);
            }
            t = jacobian_double(t);
            k_s = k_s >> 1;
        }
    }
    return r;
}

fn small_jacobian_mul(p: JacobianPoint, k: u32) -> JacobianPoint {
    var r: JacobianPoint = JACOBIAN_IDENTITY;
    var t: JacobianPoint = p;
    var k_s = k;
    for (var j = 0u; j < W; j = j + 1u) {
        if ((k_s & 1) == 1u) {
            r = jacobian_add(r, t);
        }
        t = jacobian_double(t);
        k_s = k_s >> 1;
    }
    return r;
}

