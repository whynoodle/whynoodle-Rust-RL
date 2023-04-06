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
    // last_turn: (board before last own move, allowed moves, NN output, move choosen from NN)
    last_turn: (Array2<f32>, Array1<f32>, Array1<f32>, usize),
    replay_buffer: ReplayBuffer<Array2<f32>>,
    rng: ThreadRng,
}

impl DQlearning {
    // TODO add mini_batch_size to bs, so that bs % mbs == 0
    pub fn new(exploration: f32, batch_size: usize, mut nn: NeuralNetwork, use_ddqn: bool) -> Self {
        if nn.get_batch_size() % batch_size != 0 {
            eprintln!(
                "not implemented yet, unsure how to store 
                intermediate vals before weight updates"
            );
            unimplemented!();
        }
        nn.set_batch_size(batch_size);
        let target_nn = if use_ddqn {
            nn.clone()
        } else {
            NeuralNetwork::new1d(0, "none".to_string(), "none".to_