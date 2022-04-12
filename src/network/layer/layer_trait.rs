use ndarray::ArrayD;

/// Layer Interface:  
/// All layers passed to the neural network must implement this trait
///
pub trait Layer: Send + Sync {
    /// A unique String to identify the layer type, e.g. "Dense" or "Flatten"
    ///
    fn get_type(&self) -> String;

    /// The number of trainable parameters in this Layer.  
    /// Might be zero for some layers like "Flatten".
    ///
    fn get_num_parameter(&self) -> usize;

    /// Each layer is required to predict is output shape given the input shape.
    ///
    fn get_output_shape(&self, input_dim: Vec<usize>) -> Vec<usize>;

    /// This method is used for the inference part, when no training is required.  
    /// It comes with a smaller memory footprint than the forward() method, which stores information for a following backward() call.
    ///
    fn predict(&self, input: ArrayD<f32>) -> ArrayD<f32>; // similar to forward(..), but no