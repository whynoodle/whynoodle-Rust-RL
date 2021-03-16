use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::{Array1, ArrayD};

/// A Sigmoid layer,
#[derive(Default)]
pub struct SigmoidLayer {
    output: ArrayD<f32>,
}

impl SigmoidLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        SigmoidLayer {
            output: Array1::zeros(0).into_dyn(),
        }
    }
}

impl Functional for SigmoidLayer {}

impl Layer for SigmoidLayer {
    fn get_type(&self) -> String {
        "Sigmoid Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn get_num_parameter(&self) -