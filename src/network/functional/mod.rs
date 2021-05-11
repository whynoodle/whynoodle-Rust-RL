/// This trait forces functional layers and functions to implement Send and Sync
mod functional_trait;

pub use functional_trait::Functional;

/// This module contains the most common activation functions like sigmoid, relu, or softmax.
pub mod activation_layer;

/// This submodule provides various error function.
///
/// Beside of the normal forward() and backward() functions some Error functions implement both functions in a *_from_logits() variant.  
/// They merge the