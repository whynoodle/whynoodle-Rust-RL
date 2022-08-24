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
    fn default() -> Self {
        Adam::new(0.9, 0.999)
    }
}

impl Adam {
    /// Common values for beta1 and beta2 are 0.9 and 0.999.
    pub fn new(beta1: f32, beta2: f32) -> Self {
        Adam {
            previous_sum: Array::zeros(0).into_dyn(),
            previous_sum_squared: Array::zeros(0).into_dyn(),
            beta1,
            beta2,
            t: 1.,
        }
    }
}

impl Optimizer for Adam {
    fn get_type(&self) -> String {
        format!("Adam")
    }
    fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.previous_sum = Array::zeros(shape.clone());
        self.previous_sum_squared = Array::zeros(shape);
    }
    fn optimize(&mut self, delta_w: ArrayD<f32>) -> ArrayD<f32> {
        self.previous_sum = &self.previous_sum * self.beta1 + delta_w.clone() * (1. - self.beta1);
        self.previo