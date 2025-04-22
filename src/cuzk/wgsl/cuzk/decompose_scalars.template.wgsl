{{> structs }}
{{> bigint_funcs }}
{{> field_funcs }}
{{> montgomery_product_funcs }}
{{> extract_word_from_bytes_le_funcs }}

/// Input storage buffers.
@group(0) @binding(0)
var<storage, read> scalars: array<u32>;

/// Output storage buffers.
@group(0) @binding(1)
var<storage, read_write> chunks: array<u32>;

/// Uniform storage buffer.
@group(0) @binding(2)
var<uniform> input_size: u32;

const NUM_SUBTASKS = {{ num_subtasks }}u;

/// Scalar chunk bitwidth.
const CHUNK_SIZE = {{ chunk_size }}u;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x; 
    let gidy = global_id.y; 
    let id = gidx * {{ num_y_workgroups }} + gidy;

    let INPUT_SIZE = input_size;
    /// Decompose scalars.
    var scalar_bytes: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) {
        let s = scalars[id * 8 + i];
        let hi = s >> 16u;
        let lo = s & 65535u;
        scalar_bytes[15 - (i * 2)] = lo;
        scalar_bytes[15 - (i * 2) - 1] = hi;
    }

    /// Extract scalar chunks and store them in chunks_arr.
    var chunks_arr: array<u32, {{ num_subtasks }}>;
    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        let offset = i * INPUT_SIZE;
        chunks_arr[i] = extract_word_from_bytes_le(scalar_bytes, i, CHUNK_SIZE);
    }
    let div = i32(NUM_SUBTASKS) * i32(CHUNK_SIZE) - 256 + 16 - i32(CHUNK_SIZE);
    if (div >= 0) {
        chunks_arr[NUM_SUBTASKS - 1u] = scalar_bytes[0] >> u32(div);
    }

    /// Iterate through chunks_arr to compute the signed indices.
    let l = {{ num_columns }}u;
    let s = l / 2u;

    var signed_slices: array<i32, {{ num_subtasks }}>;
    var carry = 0u;
    for (var i = 0u; i < NUM_SUBTASKS; i ++) {
        signed_slices[i] = i32(chunks_arr[i] + carry);
        if (signed_slices[i] >= i32(s)) {
            signed_slices[i] = (i32(l) - signed_slices[i]) * -1i;
            carry = 1u;
        } else {
            carry = 0u;
        }
    }

    for (var i = 0u; i < NUM_SUBTASKS; i++) {
        let offset = i * INPUT_SIZE;

        /// Note that we add s (half_num_columns) to the bucket index so we
        /// don't store negative values, while retaining information about the
        /// sign of the original index.
        chunks[id + offset] = u32(signed_slices[i]) + s;
    }

    {{{ recompile }}}
}
