mod convolution;
mod dense;
mod dropout;
mod flatten;
//pub mod conv_test;
mod layer_trait;

/// This trait defines all functions which a layer has to implement to be used as a part of the neural network.
pub use layer_trait::Layer;

/// This layer implements a classical convolution layer.
pub use convolution::ConvolutionLayer2D;
/// This lay