use handlebars::Handlebars;
use once_cell::sync::Lazy;
use serde_json::json;

// Templates
pub static DECOMPOSE_SCALARS_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/cuzk/decompose_scalars.template.wgsl").to_string()
});
pub static EXTRACT_WORD_FROM_BYTES_LE_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/cuzk/extract_word_from_bytes_le.template.wgsl").to_string()
});
pub static MONTGOMERY_PRODUCT_FUNCS_2: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/montgomery/mont_product.template.wgsl").to_string()
});
pub static MONTGOMERY_PRODUCT_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/montgomery/mont_pro_product.template.wgsl").to_string()
});
pub static BARRETT_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/field/barrett.template.wgsl").to_string()
});
pub static MONTGOMERY_PRODUCT_FUNCS_CIOS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/montgomery/mont_pro_cios.template.wgsl").to_string()
});
pub static EC_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/curve/ec.template.wgsl").to_string()
});
pub static FIELD_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/field/field.template.wgsl").to_string()
});
pub static BIGINT_FUNCS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/bigint/bigint.template.wgsl").to_string()
});
pub static STRUCTS: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/struct/structs.template.wgsl").to_string()
});

// Shaders
pub static TRANSPOSE_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/cuzk/transpose.template.wgsl").to_string()
});
pub static SMVP_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/cuzk/smvp.template.wgsl").to_string()
});
pub static BPR_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/cuzk/bpr.template.wgsl").to_string()
});
pub static TEST_FIELD_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/test/test_field.wgsl").to_string()
});
pub static TEST_DIFF_IMPL_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/test/test_diff_impl.wgsl").to_string()
});
pub static TEST_POINT_SHADER: Lazy<String> = Lazy::new(|| {
    include_str!("wgsl/test/test_point.wgsl").to_string()
});

use crate::cuzk::utils::{calc_bitwidth, gen_mu_limbs, gen_p_limbs};

use super::{msm::{P, PARAMS}, utils::{gen_p_limbs_plus_one, gen_r_limbs, gen_zero_limbs}};
pub struct ShaderManager {
    word_size: usize,
    chunk_size: usize,
    input_size: usize,
    num_words: usize,
    mask: usize,
    index_shift: usize,
    p_limbs: String,
    p_limbs_plus_one: String,
    zero_limbs: String,
    r_limbs: String,
    slack: usize,
    w_mask: usize,
    n0: u32,
    mu_limbs: String,
}

impl ShaderManager {
    pub fn new(word_size: usize, chunk_size: usize, input_size: usize) -> Self {
        let p_bit_length = calc_bitwidth(&P); 
        let num_words = PARAMS.num_words;
        let r = PARAMS.r.clone();
        println!("P: {:?}", P);
        println!("P limbs: {}", gen_p_limbs(&P, num_words, word_size));
        println!("W_MASK: {:?}", (1 << word_size) - 1);
        println!("R limbs: {}", gen_r_limbs(&r, num_words, word_size));
        Self {
            word_size,
            chunk_size,
            input_size,
            num_words,
            mask: 1 << word_size - 1,
            index_shift: 1 << (chunk_size - 1),
            p_limbs: gen_p_limbs(&P, num_words, word_size),
            p_limbs_plus_one: gen_p_limbs_plus_one(&P, num_words, word_size),
            zero_limbs: gen_zero_limbs(num_words),
            slack: num_words * word_size - p_bit_length,
            w_mask: (1 << word_size) - 1,
            n0: PARAMS.n0.clone(),
            r_limbs: gen_r_limbs(&r, num_words, word_size),
            mu_limbs: gen_mu_limbs(&P, num_words, word_size),
        }
    }

    pub fn gen_transpose_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_string("transpose", TRANSPOSE_SHADER.as_str())
            .unwrap();
        let data = json!({
            "workgroup_size": workgroup_size,
        });
        // TODO: Add recompile
        handlebars.render("transpose", &data).unwrap()
    }

    pub fn gen_smvp_shader(&self, workgroup_size: usize, num_csr_cols: usize) -> String {
        println!("num_csr_cols: {:?}", num_csr_cols);
        println!("workgroup_size: {:?}", workgroup_size);
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_string("smvp", SMVP_SHADER.as_str())
            .unwrap();

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("ec_funcs", EC_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();

        let data = json!({
            "word_size": self.word_size,
            "num_words": self.num_words,
            "num_columns": num_csr_cols,
            "workgroup_size": workgroup_size,
            "n0": self.n0,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "index_shift": self.index_shift,
            "half_num_columns": num_csr_cols / 2,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        // TODO: Add recompile
        handlebars.render("smvp", &data).unwrap()
    }

    pub fn gen_bpr_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_string("bpr", BPR_SHADER.as_str())
            .unwrap();

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("ec_funcs", EC_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();
        let data = json!({
            "workgroup_size": workgroup_size,
            "word_size": self.word_size,
            "num_words": self.num_words,
            "n0": self.n0,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "index_shift": self.index_shift,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        // TODO: Add recompile
        handlebars.render("bpr", &data).unwrap()
    }

    pub fn gen_decomp_scalars_shader(
        &self,
        workgroup_size: usize,
        num_y_workgroups: usize,
        num_subtasks: usize,
        num_columns: usize,
    ) -> String {
        println!("num_columns: {:?}", num_columns);
        println!("num_y_workgroups: {:?}", num_y_workgroups);
        println!("num_subtasks: {:?}", num_subtasks);
        println!("workgroup_size: {:?}", workgroup_size);
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_string("decomp_scalars", DECOMPOSE_SCALARS_SHADER.as_str())
            .unwrap();

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string(
            "extract_word_from_bytes_le_funcs",
            EXTRACT_WORD_FROM_BYTES_LE_FUNCS.as_str(),
        ).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();
        let data = json!({
            "workgroup_size": workgroup_size,
            "word_size": self.word_size,
            "chunk_size": self.chunk_size,
            "num_words": self.num_words,
            "num_y_workgroups": num_y_workgroups,
            "num_subtasks": num_subtasks,
            "num_columns": num_columns,
            "n0": self.n0,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "slack": self.slack,
            "w_mask": self.w_mask,
            "mask": self.mask,
            "index_shift": self.index_shift,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "r_limbs": self.r_limbs,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        // TODO: Add recompile
        handlebars.render("decomp_scalars", &data).unwrap()
    }

    // Test methods
    pub fn gen_test_field_shader(&self) -> String {
        let mut handlebars = Handlebars::new();
        handlebars.register_template_string("test_field", TEST_FIELD_SHADER.as_str()).unwrap();
        

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs_2", MONTGOMERY_PRODUCT_FUNCS_2.as_str()).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();

        let data = json!({
            "word_size": self.word_size,
            "num_words": self.num_words,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "n0": self.n0,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        handlebars.render("test_field", &data).unwrap()
    }

    pub fn gen_test_diff_impl_shader(&self) -> String {
        let mut handlebars = Handlebars::new();
        handlebars.register_template_string("test_diff_impl", TEST_DIFF_IMPL_SHADER.as_str()).unwrap();
        

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs_2", MONTGOMERY_PRODUCT_FUNCS_2.as_str()).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();

        let data = json!({
            "word_size": self.word_size,
            "num_words": self.num_words,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "n0": self.n0,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        handlebars.render("test_diff_impl", &data).unwrap()
    }

    pub fn gen_test_point_shader(&self) -> String {
        let mut handlebars = Handlebars::new();
        handlebars.register_template_string("test_point", TEST_POINT_SHADER.as_str()).unwrap();
        

        handlebars.register_template_string("structs", STRUCTS.as_str()).unwrap();
        handlebars.register_template_string("bigint_funcs", BIGINT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("field_funcs", FIELD_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("ec_funcs", EC_FUNCS.as_str()).unwrap();
        handlebars.register_template_string("barrett_funcs", BARRETT_FUNCS.as_str()).unwrap();
        let data = json!({
            "word_size": self.word_size,
            "num_words": self.num_words,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "n0": self.n0,
            "mu_limbs": self.mu_limbs,
            "slack": self.slack,
        });
        handlebars.render("test_point", &data).unwrap()
    }
}
