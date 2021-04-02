use super::Error;
use ndarray::{Array, ArrayD};

use ndarray_stats::QuantileExt;

/// This implements the categorical crossentropy loss.
#[derive(Clone, Default)]
pub struct CategoricalCrossEntropyError {}

impl CategoricalCrossEntropyError {
    /// No parameters required.
    pub fn new() -> Self {
        CategoricalCrossEntropyError {}
    }

    fn clip_values(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        arr.mapv_inplace(|x| if x > 0.9999 { 0.9999 } else { x });
        arr.mapv(|x| if x < 1e-8 { 1e-8 } else { x })
    }
}

impl Error for CategoricalCrossEntropyError {
    fn get_type(&self) -> String {
        format!("Categorical Crossentropy")
    }

    fn forward(&self, mut output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        output = self.clip_values(output);
        let loss = -(target * output.mapv(f32::ln)).sum();
        Array::from_elem(1, loss).into_dyn()
    }

    fn backward(&self, outpu