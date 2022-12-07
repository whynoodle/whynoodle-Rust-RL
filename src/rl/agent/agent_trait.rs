use ndarray::{Array1, Array2};

/// A trait including all functions required to train them.
pub trait Agent {
    /// Returns a simple string identifying the specific agent type.
    fn get_id(&self) -> String;

    /// Expect the agent to return a single usize value corresponding to a (legal) action he picked.
    ///
    /// The concrete encoding of actions as usize value has to be looked up in the documentation of the specific environment.  
    /// Advanced agents shouldn't need knowledge about the used encoding.
    fn get_move(&mut self, env: Array2<f32>, actions: Array1<bool>, reward: f32) -> usize;

    /// Informs the agent that the current epoch has finished and tells him about hi