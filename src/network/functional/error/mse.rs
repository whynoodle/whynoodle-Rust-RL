use super::Error;
use ndarray::{Array1, ArrayD, Ix1};

/// This function calculates the mean of squares of errors between the neural network output and the ground truth.
#[deriv