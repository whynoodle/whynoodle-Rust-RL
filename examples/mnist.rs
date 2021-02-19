use datasets::mnist;
use ndarray::{Array2, Array4, Axis};
use rand::Rng;
use rust_rl::network::nn::NeuralNetwork;
use std::time::Instant;

fn new() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new3d((1, 28, 28), "cce".to_string(), "adam".to_string());
    nn.set_batch_size(32);
    nn.set_learning_rate(1e-3);
    nn.add_convolution((3, 3), 10, 1);
    nn.add_flatten();
    nn.add_activation("sigmoid");
    nn.add_dropout(0.5);
    nn.add_dense(10);
    nn.add_activation("softmax");
    nn
}

fn test(nn: &mut NeuralNetwork, input: &Array4<f32>, feedback: 