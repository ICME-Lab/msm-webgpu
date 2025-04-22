{{> structs }}
{{> montgomery_product_funcs }}
{{> field_funcs }}
{{> bigint_funcs }}

{{> ec_funcs }}

@group(0) @binding(0)
var<storage, read_write> a: Point;
@group(0) @binding(1)
var<storage, read_write> b: Point;
@group(0) @binding(2)
var<storage, read_write> result: Point;

@compute @workgroup_size(1)
fn test_point_add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    result = point_add(a, b);
}


// @compute @workgroup_size(1)
// fn test_field_add(
//     @builtin(global_invocation_id) global_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>
// ) {
//     var x = a;
//     var y = b;
//     result = field_add(&x, &y);
// }

// @compute @workgroup_size(1)
// fn test_field_sub(
//     @builtin(global_invocation_id) global_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>
// ) {
//     var x = a;
//     var y = b;
//     result = field_sub(&x, &y);   
// }