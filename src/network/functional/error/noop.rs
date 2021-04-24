use super::Error;
use ndarray::{Array1, ArrayD};

/// This function returns 42 during the forward call and forwards the ground trouth unchanged to the previous layer.
///
/// It is intended for debug purpose only.
#[derive(Clone, Default)]
pub struct NoopError {}

impl NoopError {
    /// No parameters required.
    pub fn new() -> Self {
   