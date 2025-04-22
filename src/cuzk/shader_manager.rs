use std::collections::BTreeMap;

use handlebars::Handlebars;
use num_bigint::{BigInt, BigUint};
use num_traits::{Num, One};
use serde_json::json;

// Templates
pub const EXTRACT_WORD_FROM_BYTES_LE_FUNCS: &str =
    "src/cuzk/wgsl/cuzk/extract_word_from_bytes_le.template.wgsl";
// pub const MONTGOMERY_PRODUCT_FUNCS: &str = "src/cuzk/wgsl/montgomery/mont_pro_product.template.wgsl";
pub const MONTGOMERY_PRODUCT_FUNCS: &str = "src/cuzk/wgsl/montgomery/mont_product.template.wgsl";
pub const EC_FUNCS: &str = "src/cuzk/wgsl/curve/ec.template.wgsl";
pub const FIELD_FUNCS: &str = "src/cuzk/wgsl/field/field.template.wgsl";
pub const BIGINT_FUNCS: &str = "src/cuzk/wgsl/bigint/bigint.template.wgsl";
pub const STRUCTS: &str = "src/cuzk/wgsl/struct/structs.template.wgsl";

// Shaders
pub const TRANSPOSE_SHADER: &str = "src/cuzk/wgsl/cuzk/transpose.template.wgsl";
pub const SMVP_SHADER: &str = "src/cuzk/wgsl/cuzk/smvp.template.wgsl";
pub const BPR_SHADER: &str = "src/cuzk/wgsl/cuzk/bpr.template.wgsl";
pub const DECOMPOSE_SCALARS_SHADER: &str = "src/cuzk/wgsl/cuzk/decompose_scalars.template.wgsl";
pub const TEST_FIELD_SHADER: &str = "src/cuzk/wgsl/field/test_field.wgsl";

use crate::cuzk::utils::gen_p_limbs;

use super::{msm::{P, PARAMS}, utils::{gen_p_limbs_plus_one, gen_r_limbs, gen_zero_limbs}};
pub struct ShaderManager {
    word_size: usize,
    chunk_size: usize,
    input_size: usize,
    num_words: usize,
    mask: usize,
    index_shift: usize,
    two_pow_word_size: usize,
    two_pow_chunk_size: usize,
    p_limbs: String,
    p_limbs_plus_one: String,
    zero_limbs: String,
    r_limbs: String,
    p_bit_length: usize,
    slack: usize,
    w_mask: usize,
    n0: u32,
}

impl ShaderManager {
    pub fn new(word_size: usize, chunk_size: usize, input_size: usize) -> Self {
        let p_bit_length = 254; // TODO: Parameterise
        let num_words = PARAMS.num_words;
        let r = PARAMS.r.clone();
        Self {
            word_size,
            chunk_size,
            input_size,
            num_words,
            mask: 1 << word_size - 1,
            index_shift: 1 << (chunk_size - 1),
            two_pow_word_size: 1 << word_size,
            two_pow_chunk_size: 1 << chunk_size,
            p_limbs: gen_p_limbs(&P, num_words, word_size),
            p_limbs_plus_one: gen_p_limbs_plus_one(&P, num_words, word_size),
            zero_limbs: gen_zero_limbs(num_words),
            p_bit_length,
            slack: num_words * word_size - p_bit_length,
            w_mask: (1 << word_size) - 1,
            n0: PARAMS.n0.clone(),
            r_limbs: gen_r_limbs(&r, num_words, word_size)
        }
    }

    pub fn gen_transpose_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("transpose", TRANSPOSE_SHADER)
            .unwrap();
        let data = json!({
            "workgroup_size": workgroup_size,
        });
        // TODO: Add recompile
        handlebars.render("transpose", &data).unwrap()
    }

    pub fn gen_smvp_shader(&self, workgroup_size: usize, num_csr_cols: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("smvp", SMVP_SHADER)
            .unwrap();

        handlebars.register_template_file("structs", STRUCTS).unwrap();
        handlebars.register_template_file("bigint_funcs", BIGINT_FUNCS).unwrap();
        handlebars.register_template_file("ec_funcs", EC_FUNCS).unwrap();
        handlebars.register_template_file("field_funcs", FIELD_FUNCS).unwrap();
        handlebars.register_template_file("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS).unwrap();

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
            "two_pow_word_size": self.two_pow_chunk_size,
            "index_shift": self.index_shift,
            "half_num_columns": num_csr_cols / 2,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
        });
        // TODO: Add recompile
        handlebars.render("smvp", &data).unwrap()
    }

    pub fn gen_bpr_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("bpr", BPR_SHADER)
            .unwrap();

        handlebars.register_template_file("structs", STRUCTS).unwrap();
        handlebars.register_template_file("bigint_funcs", BIGINT_FUNCS).unwrap();
        handlebars.register_template_file("ec_funcs", EC_FUNCS).unwrap();
        handlebars.register_template_file("field_funcs", FIELD_FUNCS).unwrap();
        handlebars.register_template_file("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS).unwrap();

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
            "two_pow_word_size": self.two_pow_chunk_size,
            "index_shift": self.index_shift,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
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
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("decomp_scalars", DECOMPOSE_SCALARS_SHADER)
            .unwrap();

        handlebars.register_template_file("structs", STRUCTS).unwrap();
        handlebars.register_template_file("bigint_funcs", BIGINT_FUNCS).unwrap();
        handlebars.register_template_file("field_funcs", FIELD_FUNCS).unwrap();
        handlebars.register_template_file("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS).unwrap();
        handlebars.register_template_file(
            "extract_word_from_bytes_le_funcs",
            EXTRACT_WORD_FROM_BYTES_LE_FUNCS,
        ).unwrap();

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
            "two_pow_word_size": self.two_pow_word_size,
            "two_pow_chunk_size": self.two_pow_chunk_size,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "r_limbs": self.r_limbs
        });
        // TODO: Add recompile
        handlebars.render("decomp_scalars", &data).unwrap()
    }

    // Test methods
    pub fn gen_test_field_shader(&self) -> String {
        let mut handlebars = Handlebars::new();
        handlebars.register_template_file("test_field", TEST_FIELD_SHADER).unwrap();
        

        handlebars.register_template_file("structs", STRUCTS).unwrap();
        handlebars.register_template_file("bigint_funcs", BIGINT_FUNCS).unwrap();
        handlebars.register_template_file("field_funcs", FIELD_FUNCS).unwrap();
        handlebars.register_template_file("montgomery_product_funcs", MONTGOMERY_PRODUCT_FUNCS).unwrap();

        let data = json!({
            "word_size": self.word_size,
            "num_words": self.num_words,
            "p_limbs": self.p_limbs,
            "p_limbs_plus_one": self.p_limbs_plus_one,
            "zero_limbs": self.zero_limbs,
            "r_limbs": self.r_limbs,
            "mask": self.mask,
            "w_mask": self.w_mask,
            "two_pow_word_size": self.two_pow_word_size,
            "two_pow_chunk_size": self.two_pow_chunk_size,
            "num_words_mul_two": self.num_words * 2,
            "num_words_plus_one": self.num_words + 1,
            "n0": self.n0,
        });
        handlebars.render("test_field", &data).unwrap()
    }
}
