use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Clone)]
pub struct Adam {
    previous_sum: ArrayD<f32>,
    previous_sum_squared: ArrayD<f32>,
    beta1: f32,
    beta2: f32,
    t: f32,
}

impl Default for Adam {
    fn default()