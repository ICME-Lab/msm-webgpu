{{> structs }}
{{> montgomery_product_funcs }}
{{> montgomery_product_funcs_2 }}
{{> field_funcs }}
{{> bigint_funcs }}

@group(0) @binding(0)
var<storage, read_write> a: BigInt;
@group(0) @binding(1)
var<storage, read_write> b: BigInt;
@group(0) @binding(2)
var<storage, read_write> result_1: BigInt;
@group(0) @binding(3)
var<storage, read_write> result_2: BigInt;

@compute @workgroup_size(1)
fn test_diff_impl(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    var x = a;
    var y = b;
    result_1 = montgomery_product(&x, &y);
    result_2 = montgomery_product_2(&x, &y);
}

