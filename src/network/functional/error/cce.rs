use super::Error;
use ndarray::{Array, ArrayD};

use ndarray_stats::QuantileExt;

/// This implements the categorical crossentropy loss.
#[derive(Clone, Default)]
pub struct CategoricalCrossEntropyError {}

impl