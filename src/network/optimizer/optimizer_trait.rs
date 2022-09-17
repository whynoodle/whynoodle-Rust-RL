use ndarray::{Array1, Array2, Array3, ArrayD};

/// A trait defining functions to alter the weight and bias updates before they are applied.
///
/// All neural network layers are expected to call the coresponding functions after calculating the deltas   
/// and only to apply the results of these functions to update their weights or biases.
pub trait Optimizer: Send + Sync {
    ///
    fn set_input_shape(&mut self, shape: Vec<usize>);
    /// Returns a string identifying the specific optimizer type. Examples are "Adam", "Momentum", 