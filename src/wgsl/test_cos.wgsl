struct Array {
data: array<f32>,
}; 

@group(0) 
@binding(0)
var<storage, read> x: Array;

@group(0) 
@binding(1)
var<storage, read_write> y: Array;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
let gidx = global_id.x;
y.data[gidx] = cos(x.data[gidx]);
}