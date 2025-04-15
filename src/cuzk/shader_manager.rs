use std::collections::BTreeMap;

use handlebars::Handlebars;

pub struct ShaderManager {
    word_size: usize,
    chunk_size: usize,
    input_size: usize,
    num_words: usize,
}

impl ShaderManager {
    pub fn new(word_size: usize, chunk_size: usize, input_size: usize, num_words: usize) -> Self {
        Self {
            word_size,
            chunk_size,
            input_size,
            num_words,
        }
    }

    pub fn gen_transpose_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("transpose", "src/cuzk/wgsl/transpose.wgsl")
            .unwrap();
        let mut data = BTreeMap::new();
        data.insert("workgroup_size", workgroup_size);
        // TODO: Add recompile
        handlebars.render("transpose", &data).unwrap()
    }

    pub fn gen_smvp_shader(&self, workgroup_size: usize, num_csr_cols: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("smvp", "src/cuzk/wgsl/smvp.wgsl")
            .unwrap();
        let mut data = BTreeMap::new();
        data.insert("workgroup_size", workgroup_size);
        data.insert("num_columns", num_csr_cols);
        data.insert("word_size", self.word_size);
        data.insert("num_words", self.num_words);
        // TODO: Add recompile
        handlebars.render("smvp", &data).unwrap()
    }

    pub fn gen_bpr_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("bpr", "src/cuzk/wgsl/bpr.wgsl")
            .unwrap();
        let mut data = BTreeMap::new();
        data.insert("workgroup_size", workgroup_size);
        data.insert("word_size", self.word_size);
        data.insert("num_words", self.num_words);
        // TODO: Add recompile
        handlebars.render("bpr", &data).unwrap()
    }

    pub fn gen_decomp_scalars_shader(&self, workgroup_size: usize) -> String {
        let mut handlebars = Handlebars::new();
        handlebars
            .register_template_file("decomp_scalars", "src/cuzk/wgsl/decomp_scalars.wgsl")
            .unwrap();
        let mut data = BTreeMap::new();
        data.insert("workgroup_size", workgroup_size);
        data.insert("word_size", self.word_size);
        data.insert("num_words", self.num_words);
        // TODO: Add recompile
        handlebars.render("decomp_scalars", &data).unwrap()
    }
}
