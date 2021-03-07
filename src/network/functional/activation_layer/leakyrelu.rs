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

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn get_num_parameter(&self) -> usize {
        0
    }

    fn predict(&self, x: ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|x| if x > 0. { x } else { 0.01