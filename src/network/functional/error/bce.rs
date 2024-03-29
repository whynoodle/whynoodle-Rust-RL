
use super::Error;
use crate::network::functional::activation_layer::SigmoidLayer;
use crate::network::layer::Layer;
use ndarray::{Array, ArrayD};

/// This implements the binary crossentropy.
pub struct BinaryCrossEntropyError {
    activation_function: Box<dyn Layer>,
}

impl Default for BinaryCrossEntropyError {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BinaryCrossEntropyError {
    fn clone(&self) -> Self {
        BinaryCrossEntropyError::new()
    }
}

impl BinaryCrossEntropyError {
    /// No parameters required.
    pub fn new() -> Self {
        BinaryCrossEntropyError {
            activation_function: Box::new(SigmoidLayer::new()),
        }
    }
}

unsafe impl Sync for BinaryCrossEntropyError {}
unsafe impl Send for BinaryCrossEntropyError {}

impl Error for BinaryCrossEntropyError {
    fn get_type(&self) -> String {
        format!("Binary Crossentropy")
    }

    // loss after activation function (which probably was sigmoid)
    fn forward(&self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -target.clone() * input.mapv(f32::ln) - (1. - target) * input.mapv(|x| (1. - x).ln())
    }

    // deriv after activation function (which probably was sigmoid)
    fn backward(&self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        -&target / &input + (1.0 - target) / (1.0 - input)
    }

    // takes input from last dense/conv/.. layer directly, without activation function in between
    //Loss(t,z) = max(z,0) - tz + log(1+ e^(-|z|)), t is label
    fn loss_from_logits(&self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        let tmp = input.mapv(|z| 1. + (-f32::abs(z)).exp());
        let loss = input.mapv(|x| f32::max(0., x)) + tmp.mapv(f32::ln) - input * target.clone();
        let cost: f32 = loss.sum() / target.len() as f32;
        Array::from_elem(1, cost).into_dyn()
    }

    // takes input from last dense/conv/.. layer directly, without activation function in between
    fn deriv_from_logits(&self, input: ArrayD<f32>, target: ArrayD<f32>) -> ArrayD<f32> {
        self.activation_function.predict(input) - target
    }

    fn clone_box(&self) -> Box<dyn Error> {
        Box::new(BinaryCrossEntropyError::new())
    }
}

//https://gombru.github.io/2018/05/23/cross_entropy_loss/
//https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

//numeric stable version from here (as in keras/tf):
//https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c