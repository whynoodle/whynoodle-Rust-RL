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

    fn clip_values(&self, m