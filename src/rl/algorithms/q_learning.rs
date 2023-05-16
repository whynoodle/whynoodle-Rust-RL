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
    repla