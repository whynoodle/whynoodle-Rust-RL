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
    /// The learning_rate is expected to be in the range [0,1].
    /// A batch_size of 1 basically means that no batch processing happens.
    /// A batch_size of 0, a learning_rate outside of [0,1], or an input or output dimension of 0 will result in an error.
    /// TODO: return Result<Self, Error> instead of Self
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        //xavier init
        let weights: Array2<f32> = Array::random(
            (output_dim, input_dim),
            Normal::new(0.0, 2.0 / ((output_dim + input_dim) as f32).sqrt()).unwrap(),
        );
        let bias: Array1<f32> = Array::zeros(output_dim); //https://cs231n.github.io/neural-networks-2/#init
        let mut weight_optimizer = optimizer.clone_box();
        let mut bias_optimizer = optimizer;
        weight_optimizer.set_input_shape(vec![output_dim, input_dim]);
        bias_optimizer.set_input_shape(vec![output_dim]);
        new_from_matrices(
            weights,
            bias,
            input_dim,
            output_dim,
            batch_size,
            learning_rate,
            weight_optimizer,
            bias_optimizer,
        