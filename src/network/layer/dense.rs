use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{Array, Array1, Array2, ArrayD, Axis, Ix1, Ix2};
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be better & faster
use ndarray_rand::RandomExt;

/// A dense (also called fully connected) layer.
pub struct DenseLayer {
    input_dim: usize,
    output_dim: usize,
    learning_rate: f32,
    weights: Array2<f32>,
    bias: Array1<f32>,
    net: Array2<f32>,
    feedback: Array2<f32>,
    batch_size: usize,
    forward_passes: usize,
    backward_passes: usize,
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
}

impl DenseLayer {
    /// A common constructor for a dense layer.
    ///
    /// The learning_rate is expected to be in the range [0,