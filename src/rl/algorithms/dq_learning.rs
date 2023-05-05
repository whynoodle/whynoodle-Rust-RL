use super::{Observation, ReplayBuffer};
use crate::network::nn::NeuralNetwork;
use crate::rl::algorithms::utils;
use ndarray::{par_azip, Array1, Array2, Array4};
use ndarray_stats::QuantileExt;
use rand::rngs::ThreadRng;
use rand::Rng;

static EPSILON: f32 = 1e-4;

pub struct DQlearning {
    nn: NeuralNetwork,
    use_ddqn: bool,
    target_nn: NeuralNetwork,
    target_update_counter: usize,
    target_update_every: usize,
    exploration: f32,
    discount_factor: f32,
    // last_turn: (board before last own move, allowed moves, NN output, move choosen from NN)
    last_turn: (Array2<f32>, Array1<f32>, Array1<f32>, usize),
    replay_buffer: ReplayBuffer<Array2<f32>>,
    rng: ThreadRng,
}

impl DQlearning {
    // TODO add mini_batch_size to bs, so that bs % mbs == 0
    pub fn new(exploration: f32, batch_size: usize, mut nn: NeuralNetwork, use_ddqn: bool) -> Self {
        if nn.get_batch_size() % batch_size != 0 {
            eprintln!(
                "not implemented yet, unsure how to store 
                intermediate vals before weight updates"
            );
            unimplemented!();
        }
        nn.set_batch_size(batch_size);
        let target_nn = if use_ddqn {
            nn.clone()
        } else {
            NeuralNetwork::new1d(0, "none".to_string(), "none".to_string())
        };
        let discount_factor = 0.95;
        DQlearning {
            use_ddqn,
            target_nn,
            target_update_counter: 0,
            target_update_every: 20, // update after 5 episodes (entire games)
            nn,
            exploration,
            last_turn: (
                Default::default(),
                Default::default(),
                Default::default(),
                42,
            ),
            replay_buffer: ReplayBuffer::new(batch_size, 2_000),
            discount_factor,
            rng: rand::thread_rng(),
        }
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.nn.get_learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&lr) {
            return Err("learning rate must be in [0,1]!".to_string());
        }
        self.nn.set_learning_rate(lr);
        Ok(())
    }

    pub fn get_exploration_rate(&self) -> f32 {
        self.exploration
    }

    pub fn set_exploration_rate(&mut self, e: f32) -> Result<(), String> {
        if !(0.0..=1.).contains(&e) {
            return Err("exploration rate must be in [0,1]!".to_string());
        }
        self.exploration = e;
        Ok(())
    }
}

impl DQlearning {
    // learn based on last action and their result
    pub fn finish_round(&mut self, mut final_reward: f32, s1: Array2<f32>) {
        if f32::abs(final_reward) < 1e-4 {
            final_reward = 0.5; // smaller bonus for a draw
        }

        self.replay_buffer.add_memory(Observation::new(
            self.last_turn.0.clone(),
            self.last_turn.3,
            s1,
            final_reward,
            true,
        ));
        self.target_update_counter += 1;
        self.learn();
    }

    pub fn get_move(
        &mut self,
        board_arr: Array2<f32>,
        action_arr: Array1<bool>,
        reward: f32,
    ) -> usize {
        let actions = action_arr.mapv(|x| if x { 1. } else { 0. });

        // store every interesting action, as well as every 20% of the actions with zero-reward
        if f32::abs(reward) > EPSILON || self.rng.gen::<f32>() < 0.2 {
            self.replay_buffer.add_memory(Observation::new(
                self.last_turn.0.clone(),
                self.last_turn.3,
                board_arr.clone(),
                reward,
                false,
            ));
        }
        self.learn();

        let board_with_channels = board_arr
            .clone()
            .into_shape((1, board_arr.shape()[0], board_arr.shape()[1]))
            .unwrap();
        let predicted_moves = self.nn.predict3d(board_with_channels);
        let legal_predicted_moves = predicted_moves.clone() * actions.clone();
        let mut next_move = legal_predicted_moves.argmax().unwrap();

        // shall we explore a random move?
        // also select random move if predicted move not allowed
        // (e.g. legal_predicted_moves contains only 0's).
        if (self.exploration > self.rng.gen()) || (!action_arr[next_move]) {
            next_move = utils::get_random_true_entry(action_arr);
        }

        // bookkeeping
        self.last_turn = (board_arr, actions, predicted_moves, next_move);

        //println!("action: {}, \t reward: {}", self.last_turn.3, reward);

        self.last_turn.3
    }

    fn learn(&mut self) {
        if !self.replay_buffer.is_full() {
            return;
        }
        let (s0_vec, actions, s1_vec, rewards, done) = self.replay_buffer.get_memories_SoA();
        let s0_arr = vec_to_arr(s0_vec);
        let s1_arr = vec_to_arr(s1_vec);
        let done: Array1<f32> = done.mapv(|x| if !x { 1. } else { 0. });

        let current_q_list: Array2<f32> = self.nn.predict_batch(s0_arr.clone().into_dyn());
        let future_q_list_1: Array2<f32> = self.nn.predict_batch(s1_arr.clone().into_dyn());
        let future_q_list_2 = if self.use_ddqn {
            self.target_nn.predict_batch(s1_arr.into_dyn())
        } else {
            self.nn.predict_batch(s1_arr.into_dyn())
        };

        let best_future_actions: Array1<usize> = argmax(future_q_list_1);
        let future_rewards: Array1<f32> = get_future_rewards(future_q_list_2, best_future_actions);

        // TODO done vorziehen um nur bei nicht endzuständen zu predicten
        let mut new_q_list: Array1<f32> = rewards + self.discount_factor * done * future_rewards;
        new_q_list.mapv_inplace(|x| if x < 1. { x } else { 1. });
        let targets = update_targets(current_q_list, actions, new_q_list);

        self.nn.train(s0_arr.into_dyn(), targets.into_dyn());

        if self.use_ddqn && self.target_update_counter > self.target_update_every {
            self.target_nn = self.nn.clone();
            // TODO improve, only copy weights later
            // due to optimizer´s and such stuff
            self.target_update_counter = 0;
        }
    }
}

fn get_future_r