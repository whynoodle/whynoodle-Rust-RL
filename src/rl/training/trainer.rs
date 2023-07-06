
use crate::rl::agent::Agent;
use crate::rl::env::Environment;
use ndarray::Array2;

/// A trainer works on a given environment and a set of agents.
pub struct Trainer {
    env: Box<dyn Environment>,
    res: Vec<(u32, u32, u32)>,
    agents: Vec<Box<dyn Agent>>,
    learning_rates: Vec<f32>,
    exploration_rates: Vec<f32>,
    print: bool,
}

impl Trainer {
    /// We construct a Trainer by passing a single environment and one or more (possibly different) agents.
    pub fn new(
        env: Box<dyn Environment>,
        agents: Vec<Box<dyn Agent>>,
        print: bool,
    ) -> Result<Self, String> {
        if agents.is_empty() {
            return Err("At least one agent required!".to_string());
        }
        let (exploration_rates, learning_rates) = get_rates(&agents);
        Ok(Trainer {
            env,
            res: vec![(0, 0, 0); agents.len()],
            agents,
            learning_rates,
            exploration_rates,
            print,
        })
    }

    /// Returns a (#won, #draw, #lost) tripple for each agent.
    ///
    /// The numbers are accumulated over all train and bench games, either since the beginning, or the last reset_results() call.
    pub fn get_results(&self) -> Vec<(u32, u32, u32)> {
        self.res.clone()
    }

    /// Resets the (#won, #draw, #lost) values for each agents to (0,0,0).
    pub fn reset_results(&mut self) {
        self.res = vec![(0, 0, 0); self.agents.len()];
    }
