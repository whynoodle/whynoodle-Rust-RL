use crate::network::functional::Functional;
use crate::network::layer::Layer;
use ndarray::ArrayD;

/// A leaky-relu layer.
#[derive(Default)]
pub struct LeakyReLuLayer {}

impl LeakyReLuLayer {
    /// No parameters are possible.
    pub fn new() -> Self {
        Le