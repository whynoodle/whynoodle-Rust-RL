use super::{Observation, ReplayBuffer};
use crate::network::nn::NeuralNetwork;
use crate::rl::algorithms::utils;
use ndarray::{par_azip, Array1, Array2, Array4};
use ndarray_stats::QuantileExt;
use rand::rngs::ThreadRng;
use rand::Rng;

static EPSILON: f32 = 1e-4;

pub struct DQlearning {
    nn: NeuralNetwork,
    use_ddqn: bool,
    target_nn: NeuralNetwork,
    target_update_counter: usize,
    target_update_every: usize,
    exploration: f32,
    discount_factor: f32,
    // last_turn: (board before last own move, allowed moves, NN output, move choosen from N