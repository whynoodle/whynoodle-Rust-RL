use crate::rl::agent::agent_trait::Agent;
use crate::rl::algorithms::utils;
use ndarray::{Array1, Array2};

/// An agent who acts randomly.
///
/// All input is ignored except of the vector of possible actions.
/// All allowed actions are considered with an equal probability.
#[derive(Default)]
pub struct RandomAgent {}

impl RandomAgent {
    /// Returns a new instance of a random acting agent.
    pub fn new() -> Self {
        RandomAgent {}
    }
}

impl Agent for RandomAgent {
    fn get_id(&self) -> String {
    