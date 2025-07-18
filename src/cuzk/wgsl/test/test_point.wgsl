{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}
{{> barrett_funcs }}
{{> ec_funcs }}

@group(0) @binding(0)
var<storage, read_write> a: Point;
@group(0) @binding(1)
var<storage, read_write> b: Point;
@group(0) @binding(2)
var<storage, read_write> result: Point;

@group(0) @binding(3)
var<uniform> scalar: u32;

@compute @workgroup_size(1)
fn test_point_add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var r = get_r();
    var ax = a.x;
    var ay = a.y;
    var az = a.z;
    var ar_x = field_mul(&ax, &r);
    var ar_y = field_mul(&ay, &r);
    var ar_z = field_mul(&az, &r);
    var p_a = Point(ar_x, ar_y, ar_z);
    var bx = b.x;
    var by = b.y;
    var bz = b.z;
    var br_x = field_mul(&bx, &r);
    var br_y = field_mul(&by, &r);
    var br_z = field_mul(&bz, &r);
    var p_b = Point(br_x, br_y, br_z);
    result = point_add(p_a, p_b);
}

@compute @workgroup_size(1)
fn test_negate_point(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var r = get_r();    
    var ax = a.x;
    var ay = a.y;
    var az = a.z;
    var ar_x = field_mul(&ax, &r);
    var ar_y = field_mul(&ay, &r);
    var ar_z = field_mul(&az, &r);
    var p_a = Point(ar_x, ar_y, ar_z);
    result = negate_point(p_a);
}

@compute @workgroup_size(1)
fn test_double_and_add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var r = get_r();
    var ax = a.x;
    var ay = a.y;
    var az = a.z;
    var ar_x = field_mul(&ax, &r);
    var ar_y = field_mul(&ay, &r);
    var ar_z = field_mul(&az, &r);
    var p_a = Point(ar_x, ar_y, ar_z);
    result = double_and_add(p_a, scalar);
}

@compute @workgroup_size(1)
fn test_point_add_identity(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var r = get_r();
    var ax = a.x;
    var ay = a.y;
    var az = a.z;
    var ar_x = field_mul(&ax, &r);
    var ar_y = field_mul(&ay, &r);
    var ar_z = field_mul(&az, &r);
    var p_a = Point(ar_x, ar_y, ar_z);
    var p_b = POINT_IDENTITY;
    result = point_add(p_a, p_b);
}