use super::{Observation, ReplayBuffer};
use crate::rl::algorithms::utils;
use ndarray::{Array1, Array2};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::HashMap;

#[allow(dead_code)]
pub struct Qlearning {
    exploration: f32,
    learning_rate: f32,
    discount_factor: f32,
    scores: HashMap<(String, usize), f32>, // (State,Action), reward
    replay_buffer: ReplayBuffer<String>,
    last_state: String,
    last_action: usize,
    rng: ThreadRng,
    action_space_length: usize,
}

const EPSILON: f32 = 1e-4;

// based on Q-learning using a HashMap as table
//
impl Qlearning {
    pub fn new(exploration: f32, learning_rate: f32, action_space_length: usize) -> Self {
        let bs = 16;
        let discount_factor = 0.95;
        Qlearning {
            exploration,
            learning_rate,
            discount_factor,
            last_action: 42usize,
            last_state: "".to_string(),
            replay_buffer: ReplayBuffer::new(bs, 1000),
            scores: HashMap::new(),
            rng: rand::thread_rng(),
            action_space_length,
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&lr) {
            return Err("learning rate must be in [0,1]!".to_string());
        }
        self.learning_rate = lr;
        Ok(())
    }
    pub fn get_exploration_rate(&self) -> f32 {
        self.exploration
    }

    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
    