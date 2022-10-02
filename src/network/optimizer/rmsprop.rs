use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Clone)]
pub struct RMSProp {
    previous_sum_squared: ArrayD<f32>,
    decay_rate: f32,
}

impl RMSProp {
    /// A constructor including the decay rate. A common value is 0.9.
    pub fn new(decay_rate: f32) -> Sel