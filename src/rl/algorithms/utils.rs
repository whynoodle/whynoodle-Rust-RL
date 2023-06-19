use ndarray::Array1;
use rand::Rng;
pub fn get_random_true_entry(actions: Array1<bool>) -> usize {
    let num_legal_actions = actions.fold(0, |sum, &val| if val { sum + 1 } else { sum });
    assert!(num_legal_actio