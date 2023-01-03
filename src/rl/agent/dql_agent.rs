use super::results::RunningResults;
use crate::network::nn::NeuralNetwork;
use crate::rl::agent::Agent;
use crate::rl::algorithms::DQlearning;
use ndarray::{Array1, Array2};

/// An agent using Deep-Q-Learning, based on a small neural network.
pub struct DQLAgent {
    dqlearning: DQlearning,
    results: RunningResults,
}

// based on Q-learning using a HashMap as table
//
impl DQLAgent {
    /// A constructor including an initial exploration rate.
    pub fn new(exploration: f32, batch_size: usize, nn: NeuralNetwork) -> Self {
        DQLAgent {
            results: RunningResults::new(100, true),
            dqlearning: DQlearning::new(exploration, batch_size, nn, fa