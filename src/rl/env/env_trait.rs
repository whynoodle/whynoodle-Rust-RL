use ndarray::{Array1, Array2};

/// This trait defines all functions on which agents and other user might depend.
pub trait Environment {
    /// The central function which causes the environment to pass various information to the agent.
    ///
    /// The Array2 encodes the environment (the board).  
    /// The array1 encodes actions as true (allowed) or fa