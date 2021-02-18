use datasets::mnist;
use ndarray::{Array2, Array4, Axis};
use rand::Rng;
use rust_rl::network::nn::NeuralNetwork;
use std::time::Instant;

fn new() -> NeuralNetwork {
    let mut nn = N