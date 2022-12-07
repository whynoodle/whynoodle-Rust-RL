use ndarray::{Array1, Array2};

/// A trait including all functions required to train them.
pub trait Agent {
    /// Returns a simple string identifying the specific agent type.
    fn get_id(&self) -> String;

    /// Expect the agent to return a single usize value corresponding to a (legal) action he picked.
    ///
    /// The concrete encoding of actions as usize value has to be looked up in the documentation of the specific environment.  
    /// Advanced agents shouldn't need knowledge