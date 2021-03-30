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

impl Error for CategoricalCrossEntr