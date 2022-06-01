use super::optimizer_trait::Optimizer;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

/// An optimizer for more efficient weight updates.
#[derive(Default, Clone)]
pub struct AdaGrad {
    previous_sum_squared: ArrayD<f32>,
}

impl AdaGrad {
    /// No parameters available.
    pub fn new() -> Self {
        AdaGrad {
            previous_sum_squared: Array::zeros(0).into_dyn(),
        }
    }
}

impl Optimizer for AdaGrad {
    fn get_type(&self) -> String {
        format!("AdaGrad")
    }
    fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.previous_sum_squared = Array::zeros(shape);
    }
    fn optimize(&mut self, delta_w: ArrayD<f32>) -> Array