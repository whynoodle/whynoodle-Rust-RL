use crate::network::layer::Layer;
use ndarray::ArrayD;

/// A flatten layer which turns higher dimensional input into a one dimension.
///
/// A one dimensional input remains unchanged.
pub struct FlattenLayer {
    input_ndim: usize,
    input_shape: Vec<usize>,
    batch_input_shape: Vec<usize>,
    num_elements: usize,
}

impl FlattenLayer {
    /// The input_shape is re