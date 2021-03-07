use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::ArrayD;

/// A leaky-relu layer.
#[derive(Default)]
pub struct LeakyReLuLayer {}

impl LeakyReLuLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        LeakyReLuLayer {}
    }
}

impl Functional for LeakyReLuLayer {}

impl Layer for LeakyReLuLayer {
    fn get_type(&self) -> String {
        "LeakyReLu Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: 