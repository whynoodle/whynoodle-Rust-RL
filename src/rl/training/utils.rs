use std::io;

/// A helper function to create agents based on terminal input.
pub fn read_agents(n: usize) -> Vec<usize> {
    let mut agents: Vec<usize> = vec![];

    println!(
        "\nPlease insert {} numbers, seperated by whitespace, to select the agents.",
        n
    );
    prin