use ndarray::ArrayD;

/// Layer Interface:  
/// All layers passed to the neural network must implement this trait
///
pub trait Layer: Send + Syn