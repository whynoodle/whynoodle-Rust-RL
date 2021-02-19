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

fn test(nn: &mut NeuralNetwork, input: &Array4<f32>, feedback: &Array2<f32>) {
    nn.test(input.clone().into_dyn(), feedback.clone());
}

fn train(nn: &mut NeuralNetwork, num: usize, input: &Array4<f32>, fb: &Array2<f32>) {
    for _ in 0..num {
        let pos = rand::thread_rng().gen_range(0..input.shape()[0]) as usize;
        let current_input = input.index_axis(Axis(0), pos).into_owned();
        let current_fb = fb.index_axis(Axis(0), pos).into_owned();
   