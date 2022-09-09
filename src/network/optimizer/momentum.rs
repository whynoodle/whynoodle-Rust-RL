
use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Clone)]
pub struct Momentum {
    previous_delta: ArrayD<f32>,
    decay_rate: f32,
}

impl Momentum {
    /// A basic optimization over sgd.  
    /// A common value might be 0.9.
    pub fn new(decay_rate: f32) -> Self {
        Momentum {
            previous_delta: Array::zeros(0).into_dyn(),
            decay_rate,
        }
    }