use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{Array, Array1, Array2, ArrayD, Axis, Ix1, Ix2};
use ndarray_rand::rand_distr::Normal; //{StandardNo