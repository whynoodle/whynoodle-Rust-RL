use crate::network::layer::Layer;
use ndarray::{Array, ArrayD, Ix1};
pub struct ReshapeLayer {
  input_shape: [usize;3],
  num_elements: usize,
}

impl ReshapeLayer {
  pub fn new(input_shape: