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
        NoopError {}
    }
}

impl Error for NoopError {
    fn get_type(&self) -> String {
        "Noop Error function".to_string()
    }

    //printing 42 as obviously useless
    fn forward(&self, _input: ArrayD<f32>, _targ