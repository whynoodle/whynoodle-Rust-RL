
use crate::rl::env::env_trait::Environment;
use ndarray::{Array, Array1, Array2};
use std::cmp::Ordering;

static NEIGHBOURS_LIST: [&[usize]; 6 * 6] = [
    &[1, 6],
    &[0, 2, 7],
    &[1, 3, 8],
    &[2, 4, 9],
    &[3, 5, 10],
    &[4, 11],
    &[7, 0, 12],