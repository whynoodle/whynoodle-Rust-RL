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
            dqlearning: DQlearning::new(exploration, batch_size, nn, false),
        }
    }
}

impl Agent for DQLAgent {
    fn get_id(&self) -> String {
        "dqlearning agent".to_string()
    }

    fn finish_round(&mut self, reward: i8, final_state: Array2<f32>) {
        self.results.add(reward.into());
        self.dqlearning.finish_round(reward.into(), final_state);
    }

    fn get_move(&mut self, board: Array2<f32>, actions: Array1<bool>, reward: f32) -> 