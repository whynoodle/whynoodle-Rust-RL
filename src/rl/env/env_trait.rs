use ndarray::{Array1, Array2};

/// This trait defines all functions on which agents and other user might depend.
pub trait Environment {
    /// The c