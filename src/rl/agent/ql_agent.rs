use super::results::RunningResults;
use crate::rl::algorithms::Qlearning;
use ndarray::{Array1, Array2};

use crate::rl::agent::Agent;

/// An agent working on a classical q-table.
pub struct QLAgent {
    qlearning: Qlearning,
    results: Running