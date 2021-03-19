use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::{Array, ArrayD};
use ndarray_stats::QuantileExt;

/// A softmax layer.
#[derive(Default)]
pub struct SoftmaxLayer {
    output: ArrayD<f32>,
}

impl SoftmaxLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        SoftmaxLayer {
            output: Array::zeros(0).into_dyn(), //will be overwritten
        }
    }
}

impl Functional for SoftmaxLayer {}

impl Layer for SoftmaxLayer {
    fn get_type(&self) -> String {
        "Softmax Layer".to_string()
    }

    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize> {
        input_dim
    }

    fn get_num_