
use crate::network;
use ndarray::par_azip;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix1, Ix2};
use network::functional::activation_layer::{
    LeakyReLuLayer, ReLuLayer, SigmoidLayer, SoftmaxLayer,
};
use network::functional::error::{
    BinaryCrossEntropyError, CategoricalCrossEntropyError, Error, MeanSquareError, NoopError,
};
//RootMeanSquareError,
use network::layer::{ConvolutionLayer2D, DenseLayer, DropoutLayer, FlattenLayer, Layer};
use network::optimizer::*;

#[derive(Clone)]
enum Mode {
    Eval,
    Train,
}

#[derive(Clone, Default)]
struct HyperParameter {
    batch_size: usize,
    learning_rate: f32,
}

impl HyperParameter {
    pub fn new() -> Self {
        HyperParameter {
            batch_size: 1,
            learning_rate: 0.002, //10e-4
        }
    }
    pub fn batch_size(&mut self, batch_size: usize) {
        if batch_size == 0 {
            eprintln!("batch size should be > 0! Doing nothing!");
            return;
        }
        self.batch_size = batch_size;
    }
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        if learning_rate < 0. {
            eprintln!("learning rate should be >= 0! Doing nothing!");
            return;
        }
        self.learning_rate = learning_rate;
    }
    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

// Refactor in NeuralNetwork::constructor and NeuralNetwork::executor?
/// The main neural network class. It stores all relevant information.
///
/// Especially all layers, as well as their input and output shape are stored and verified.
pub struct NeuralNetwork {
    input_dims: Vec<Vec<usize>>, //each layer takes a  1 to 4-dim input. Store details here
    h_p: HyperParameter,
    layers: Vec<Box<dyn Layer>>,
    error: String, //remove due to error_function
    error_function: Box<dyn Error>,
    optimizer_function: Box<dyn Optimizer>,
    from_logits: bool,
    mode: Mode,
}

impl Clone for NeuralNetwork {
    fn clone(&self) -> NeuralNetwork {
        let new_layers: Vec<_> = self.layers.iter().map(|x| x.clone_box()).collect();
        NeuralNetwork {
            input_dims: self.input_dims.clone(),
            h_p: self.h_p.clone(),
            layers: new_layers,
            error: self.error.clone(),
            error_function: self.error_function.clone_box(),
            optimizer_function: self.optimizer_function.clone_box(),
            from_logits: self.from_logits,
            mode: self.mode.clone(),
        }
    }
}

impl NeuralNetwork {
    fn get_activation(activation_type: String) -> Result<Box<dyn Layer>, String> {
        match activation_type.as_str() {
            "softmax" => Ok(Box::new(SoftmaxLayer::new())),
            "sigmoid" => Ok(Box::new(SigmoidLayer::new())),
            "relu" => Ok(Box::new(ReLuLayer::new())),
            "leakyrelu" => Ok(Box::new(LeakyReLuLayer::new())),
            _ => Err(format!("Bad Activation Layer: {}", activation_type)),
        }
    }

    fn get_error(error_type: String) -> Result<Box<dyn Error>, String> {
        match error_type.as_str() {
            "mse" => Ok(Box::new(MeanSquareError::new())),
            //"rmse" => Ok(Box::new(RootMeanSquareError::new())),
            "bce" => Ok(Box::new(BinaryCrossEntropyError::new())),
            "cce" => Ok(Box::new(CategoricalCrossEntropyError::new())),
            "noop" => Ok(Box::new(NoopError::new())),
            _ => Err(format!("Unknown Error Function: {}", error_type)),
        }
    }

    fn get_optimizer(optimizer: String) -> Result<Box<dyn Optimizer>, String> {
        match optimizer.as_str() {
            "adagrad" => Ok(Box::new(AdaGrad::new())),
            "rmsprop" => Ok(Box::new(RMSProp::new(0.9))),
            "momentum" => Ok(Box::new(Momentum::new(0.9))),
            "adam" => Ok(Box::new(Adam::new(0.9, 0.999))),
            "none" => Ok(Box::new(Noop::new())),
            _ => Err(format!("Unknown optimizer: {}", optimizer)),
        }
    }

    fn new(input_shape: Vec<usize>, error: String, optimizer: String) -> Self {
        let error_function;
        match NeuralNetwork::get_error(error.clone()) {
            Ok(error_fun) => error_function = error_fun,
            Err(warning) => {
                eprintln!("{}", warning);
                error_function = Box::new(NoopError::new());
            }
        }
        let optimizer_function;
        match NeuralNetwork::get_optimizer(optimizer) {
            Ok(optimizer) => optimizer_function = optimizer,
            Err(warning) => {
                eprintln!("{}", warning);
                optimizer_function = Box::new(Noop::new());
            }
        }

        NeuralNetwork {
            error,
            error_function,
            optimizer_function,
            input_dims: vec![input_shape],
            layers: vec![],
            h_p: HyperParameter::new(),
            from_logits: false,
            mode: Mode::Train,
        }
    }

    /// Sets network to inference mode, dropout and backpropagation/training are disabled.
    pub fn eval_mode(&mut self) {
        self.mode = Mode::Eval;
    }

    /// Sets network to train mode, additional calculations for weight updates might occur.
    pub fn train_mode(&mut self) {
        self.mode = Mode::Train;
    }

    /// A constructor for a neural network which takes 1d input.
    pub fn new1d(input_dim: usize, error: String, optimizer: String) -> Self {
        NeuralNetwork::new(vec![input_dim], error, optimizer)
    }
    /// A constructor for a neural network which takes 2d input.
    pub fn new2d(
        (input_dim1, input_dim2): (usize, usize),
        error: String,
        optimizer: String,
    ) -> Self {
        NeuralNetwork::new(vec![input_dim1, input_dim2], error, optimizer)
    }

    /// A constructor for a neural network which takes 3d input.
    pub fn new3d(
        (input_dim1, input_dim2, input_dim3): (usize, usize, usize),
        error: String,
        optimizer: String,
    ) -> Self {
        NeuralNetwork::new(vec![input_dim1, input_dim2, input_dim3], error, optimizer)
    }

    /// A setter to adjust the optimizer.
    ///
    /// By default, batch sgd is beeing used.
    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
        self.optimizer_function = optimizer;
    }