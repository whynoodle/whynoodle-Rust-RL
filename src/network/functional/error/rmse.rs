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
        format!("Root Mean Square")
    }

    fn forward(&self, output: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let output = output.into_dimensionality::<Ix1>().unwrap();
        let target = target.into_dimensionality::<Ix1>().unwrap();
        let n = output.len() as f32;
        let err = output
            .iter()
            .zip(target.iter())
            .fold(0., |err, val| err + f32::powf(val.0 - val.1, 2.))
            / (2. * n);
        self.err = f32::sqrt(err);
        Array1::<f32>::from_elem(1, self.err).into_dyn()
    }

    fn backward(&mut self, output: ArrayD