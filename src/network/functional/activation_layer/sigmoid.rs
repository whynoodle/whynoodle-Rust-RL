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

impl Layer for SigmoidLaye