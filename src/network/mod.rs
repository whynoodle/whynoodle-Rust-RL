/// This submodule offers multiple layer implementation.
///
/// The forward and backward functions have to accept and return data in the form ArrayD\<f32>.  
/// Common activation functions are bundled under activation_layer.  
pub mod layer;

/// This submodules bundles all neural network related functionalities.
///
/// A new neural network is created with new1d(..), new2d(..), or new3d(..).  
/// For higher-dimensional input a new() function is available which accepts arbitrary sized input.  
/// The input shape, error function and the optimizer are set during network creation.  
/// If an individual error function or