use super::results::RunningResults;
use crate::rl::algorithms::Qlearning;
use ndarray::{Array1, Array2};

use crate::rl::agent::Agent;

/// An agent working on a classical q-table.
pub struct QLAgent {
    qlearning: Qlearning,
    results: RunningResults,
}

// based on Q-learning using a HashMap as table
//
impl QLAgent {
    /// A constructor with an initial exploration rate.
    pub fn new(exploration: f32, learning: f32, action_space_length: usize) -> Self {
        QLAgent {
            qlearning: Qlearning::new(exploration, learning, action_space_length),
            results: RunningResults::new(100, true),
        }
    }
}

impl Agent for QLAgent {
    fn ge