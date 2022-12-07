use ndarray::{Array1, Array2};

/// A trait including all functions required to train them.
pub trait Agent {
    /// Returns a simple string identifying the specific agent type.
    fn get_id(&self) -> String;

    /// Expect the agent to return a single usize value corresponding to a (legal) a