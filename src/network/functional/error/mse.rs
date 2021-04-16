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
    fn get_type(&self) -> String {
        format!("Mean Square")
    }

    fn forward(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let output = output.into_dimensionality::<Ix1>().unwrap();
        let target = target.into_dimensionality::<Ix1>().unwrap();
        let n = output.len() as f32;
        let err = output
            .iter()
            .zip(target.iter())
            .