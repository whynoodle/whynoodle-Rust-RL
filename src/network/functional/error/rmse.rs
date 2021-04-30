use super::Error;
use ndarray::{Array1, ArrayD, Ix1};

/// This error function works on the square-root of the mse.
#[derive(Clone, Default)]
pub struct RootMeanSquareError {
    err: f32,
}

impl RootMeanSquareError {
    /// No parameters required.
    pub fn new() -> Self {
        RootMeanSquareError { err: 0. }
    }
}

impl Error for RootMeanSquareError {
    fn get_type(&self) -> String {
   