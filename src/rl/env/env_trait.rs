use ndarray::{Array1, Array2};

/// This trait defines all functions on which agents and other user might depend.
pub trait Environment {
    /// The central function which causes the environment to pass various information to the agent.
    ///
    /// The Array2 encodes the environment (the board).  
    /// The array1 encodes actions as true (allowed) or false (illegal).
    /// The third value returns a reward for the last action of the agent. 0 before the first action of the agent.
    /// The final bool value (done) indicates, wether it is time to reset the environment.
    fn step(&self) -> (Array2<f32>, Array1<bool>, f32, bool);
    /// Update the environment based on the action given.
    ///
    /// If the action is allowed for the currently active agent then update the environm