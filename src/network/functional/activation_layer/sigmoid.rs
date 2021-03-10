use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::{Array1, ArrayD};

/// A Sigmoid layer,
#[derive(Default)]
pub struct SigmoidLayer {
    output: ArrayD<f32>,
}

impl SigmoidLayer {
    ///