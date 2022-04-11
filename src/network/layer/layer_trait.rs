use ndarray::ArrayD;

/// Layer Interface:  
/// All layers passed to the neural network must implement this trait
///
pub trait Layer: Send + Sync {
    /// A unique String to identify the layer type, e.g. "Dense" or "Flatten"
    ///
    fn get_type(&self) -> String;

    /// The number of trainable parameters in th