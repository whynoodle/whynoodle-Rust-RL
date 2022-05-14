use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Default, Clone)]
pub struct AdaGrad {
    previous_sum_squared: ArrayD<f32>,
}

impl AdaGrad {
    /// No p