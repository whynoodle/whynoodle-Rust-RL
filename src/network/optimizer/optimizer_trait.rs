use ndarray::{Array1, Array2, Array3, ArrayD};

/// A trait defining functions to alter the weight and bias updates before they are applied.
///
/// All neural network layers are expected to call the coresponding functions after calculating the deltas   
/// and only to apply the results of these functi