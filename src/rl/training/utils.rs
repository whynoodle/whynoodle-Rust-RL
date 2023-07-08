use std::io;

/// A helper function to create agents based on terminal input.
pub fn read_agents(n: usize) -> Vec<usize> {
    let mut agents: Vec<usize> = vec![];

    println!(
        "\nPlease insert {} numbers, seperated by whitespace, to select the agents.",
        n
    );
    println!("(0 for ddql, 1 for dql, 2 for ql, 3 for random, 4 for human)");
    let stdin = io::stdin();
    loop {
        let mut buffer = String::new();
        stdin.read_line(&mut buffer).unwrap();
        let nums: Vec<&str> = buffer.split(' ').collect();
        if nums.len() != n {
            println!("Please enter exactly {} values", n);
            continue;
        }
        for agent_num in nums
            .iter()
            .map(|num| usize::from_str_radix(num.trim(), 10).unwrap())
        {
            agents.push(agent_num);
        }
        break;
    }
    agents
}

/// Reads the amount of training