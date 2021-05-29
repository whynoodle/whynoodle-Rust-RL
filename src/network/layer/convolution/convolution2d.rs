
use super::conv_utils;
use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use conv_utils::*;
use ndarray::{Array, Array1, Array2, Array3, ArrayD, Axis, Ix3, Ix4};
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be cleaner & faster
use ndarray_rand::RandomExt;

/// This layer implements a convolution on 2d or 3d input.
pub struct ConvolutionLayer2D {
    batch_size: usize,
    kernels: Array2<f32>,
    in_channels: usize,
    bias: Array1<f32>, // one bias value per kernel
    padding: usize,
    last_input: ArrayD<f32>,
    filter_shape: (usize, usize),
    kernel_updates: Array2<f32>,
    bias_updates: Array1<f32>,
    learning_rate: f32,
    num_in_batch: usize,
    // Rust requires knowledge about obj size during compile time. Optimizers can be set/changed dynamically during runtime, so we just store a reference to the heap
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_shape_feedback1() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ]);
        let output = shape_into_kernel(input);
        assert_eq!(
            output,
            arr2(&[[1., 2., 3., 5., 6., 7.], [9., 10., 11., 13., 14., 15.]])
        );
    }
}

fn new_from_kernels(
    kernels: Array2<f32>,
    bias: Array1<f32>,
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
    filter_shape: (usize, usize),
    in_channels: usize,
    out_channels: usize,
    padding: usize,
    batch_size: usize,
    learning_rate: f32,
) -> ConvolutionLayer2D {
    let elements_per_kernel = filter_shape.0 * filter_shape.1 * in_channels;
    ConvolutionLayer2D {
        filter_shape,
        learning_rate,
        kernels,
        in_channels,
        padding,
        bias,
        last_input: Default::default(),
        kernel_updates: Array::zeros((out_channels, elements_per_kernel)),
        bias_updates: Array::zeros(out_channels),
        batch_size,
        num_in_batch: 0,
        weight_optimizer,
        bias_optimizer,
    }
}

impl ConvolutionLayer2D {
    /// This function prints the kernel values.
    ///
    /// It's main purpose is to analyze the learning success of the first convolution layer.
    /// Later layers might not show clear patterns.
    pub fn print_kernel(&self) {
        let n = self.kernels.nrows();
        println!("printing kernels: \n");
        for i in 0..n {
            let arr = self.kernels.index_axis(Axis(0), i);
            println!(
                "{}\n",
                arr.into_shape((self.in_channels, self.filter_shape.0, self.filter_shape.1))
                    .unwrap()
            );
        }
    }

    /// Allows setting of hand-crafted filters.
    /// 2d or 3d filters have to be reshaped into 1d, so kernels.nrows() equals the amount of kernels used.
    pub fn set_kernels(&mut self, kernels: Array2<f32>) {
        self.kernels = kernels;
    }

    /// Create a new convolution layer.
    ///
    /// Currently we only accept quadratic filter_shapes. Common dimensions are (3,3) or (5,5).
    /// The in_channels has to be set equal to the last dimension of the input images.
    /// The out_channels can be set to any positive value, 16 or 32 might be enough for simple cases or to get started.
    /// The padding will be applied to all sites of the input. Using padding: 1 on a 28x28 image will therefore result in a 30x30 input.
    pub fn new(
        filter_shape: (usize, usize),
        in_channels: usize,
        out_channels: usize,
        padding: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        assert_eq!(
            filter_shape.0, filter_shape.1,
            "currently only supporting quadratic filter!"
        );
        assert!(filter_shape.0 >= 1, "filter_shape has to be one or greater");
        assert!(