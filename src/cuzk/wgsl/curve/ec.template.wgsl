

// const POINT_IDENTITY: Point = get_paf();
const POINT_IDENTITY: Point = Point(ZERO, ONE, ZERO);

fn is_inf(p: Point) -> bool {
    return field_eq(p.z, ZERO);
}

fn point_double(p: Point) -> Point {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    var p1x = p.x;
    var p1y = p.y;
    var p1z = p.z;
    var A = montgomery_product(&p1x, &p1x);
    var B = montgomery_product(&p1y, &p1y);
    var C = montgomery_product(&B, &B);
    var X1plusB = field_add(&p1x, &B);
    var X1plusB_sq = montgomery_square(&X1plusB);
    var A_p_C = field_add(&A, &C);
    var D = field_small_scalar_shift(1, field_sub(&X1plusB_sq, &A_p_C));
    var A_shift = field_small_scalar_shift(1, A);
    var E = field_add(&A_shift, &A);
    var F = montgomery_square(&E);
    var D_shift = field_small_scalar_shift(1, D);
    var x3 = field_sub(&F, &D_shift);
    var C_shift = field_small_scalar_shift(3, C);
    var D_sub_x3 = field_sub(&D, &x3);
    var E_mul_D_sub_x3 = montgomery_product(&E, &D_sub_x3);
    var y3 = field_sub(&E_mul_D_sub_x3, &C_shift);
    var p1y_shift = field_small_scalar_shift(1, p1y);
    var z3 = montgomery_product(&p1y_shift, &p1z);
    return Point(x3, y3, z3);
}

fn point_add(p: Point, q: Point) -> Point {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
    if (field_eq(p.z, ZERO)) {
        return q;
    }
    if (field_eq(q.z, ZERO)) {
        return p;
    }
    var p1x = p.x;
    var p1y = p.y;
    var p1z = p.z;
    var q1x = q.x;
    var q1y = q.y;
    var q1z = q.z;

    var Z1Z1 = montgomery_square(&p1z);
    var Z2Z2 = montgomery_square(&q1z);
    var U1 = montgomery_product(&p1x, &Z2Z2);
    var U2 = montgomery_product(&q1x, &Z1Z1);
    var Z2Z2Z2 = montgomery_product(&Z2Z2, &q1z);
    var Z1Z1Z1 = montgomery_product(&Z1Z1, &p1z);
    var S1 = montgomery_product(&p1y, &Z2Z2Z2);
    var S2 = montgomery_product(&q1y, &Z1Z1Z1);
    if (field_eq(U1, U2)) {
        if (field_eq(S1, S2)) {
            return point_double(p);
        } else {
            return POINT_IDENTITY;
        }
    }

    var H = field_sub(&U2, &U1);
    var I = field_small_scalar_shift(2, montgomery_square(&H));
    var J = montgomery_product(&H, &I);
    var R = field_small_scalar_shift(1, field_sub(&S2, &S1));
    var V = montgomery_product(&U1, &I);
    var R_sq = montgomery_square(&R);
    var V_shift = field_small_scalar_shift(1, V);
    var J_p_V = field_add(&J, &V_shift);
    var nx = field_sub(&R_sq, &J_p_V);
    var V_sub_nx = field_sub(&V, &nx);
    var R_prod_V_sub_nx = montgomery_product(&R, &V_sub_nx);
    var shift_1_S1_J = field_small_scalar_shift(1, montgomery_product(&S1, &J));
    var ny = field_sub(&R_prod_V_sub_nx, &shift_1_S1_J);
    var Z1Z1_p_Z2Z2 = field_add(&Z1Z1, &Z2Z2);
    var p1z_p_q1z = field_add(&p1z, &q1z);
    var p1z_p_q1z_sq = montgomery_square(&p1z_p_q1z);
    var sub_p1z_p_q1z_sq_Z1Z1_p_Z2Z2 = field_sub(&p1z_p_q1z_sq, &Z1Z1_p_Z2Z2);
    var nz = montgomery_product(&H, &sub_p1z_p_q1z_sq_Z1Z1_p_Z2Z2);
    return Point(nx, ny, nz);
}

fn scalar_mul(p: Point, k: BigInt) -> Point {
    var r: Point = POINT_IDENTITY;
    var t: Point = p;
    for (var i = 0u; i < NUM_WORDS; i = i + 1u) {
        var k_s = k.limbs[i];
        for (var j = 0u; j < WORD_SIZE; j = j + 1u) {
            if ((k_s & 1) == 1u) {
                r = point_add(r, t);
            }
            t = point_double(t);
            k_s = k_s >> 1;
        }
    }
    return r;
}

/// Point negation only involves multiplying the X and T coordinates by -1 in
/// the field.
fn negate_point(point: Point) -> Point {
    var p = get_p();
    var y = point.y;
    var neg_y: BigInt;
    bigint_sub(&p, &y, &neg_y);
    return Point(point.x, neg_y, point.z);
}


fn get_paf() -> Point {
    var result: Point;
    result.x = ZERO;
    result.y = ONE;
    result.z = ZERO;
    return result;
}
/// This double-and-add code is adapted from the ZPrize test harness:
/// https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/webgpu/wgsl/Curve.ts#L78.
fn double_and_add(point: Point, scalar: u32) -> Point {
    /// Set result to the point at infinity.
    var result: Point = POINT_IDENTITY; // get_paf();

    var s = scalar;
    var temp = point;

    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = point_add(result, temp);
        }
        temp = point_double(temp);
        s = s >> 1u;
    }
    return result;
}

