use super::Error;
use ndarray::{Array1, ArrayD, Ix1};

/// This function calculates the mean of squares of errors between the neural network output and the ground truth.
#[derive(Clone, Default)]
pub struct MeanSquareError {}

impl MeanSquareError {
    /// No parameters required.
    pub fn new() -> Self {
        MeanSquareError {}
    }
}

impl Error for MeanSquareError {
    fn get_