//Author: Everett Sullivan
//Date Created: 7/09/2019
//Purpose To create a Markov Model.
//Notes:

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use rand::Rng;
use std::fmt::Debug;

const FLOAT_ERROR_TOLERANCE: f64 = 0.001;

////////////////////
//Custom Error handling code
////////////////////

#[derive(Debug)]
pub enum MMError {
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for MMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MMError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for MMError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            MMError::Syntax(ref _err_string)  => None,
        }
    }
}

////////////////////
//Markov Model code
////////////////////

#[derive(Debug)]
pub struct MarkovModel<S: Eq + Hash + Clone + Debug> {
    state_space: HashSet<S>,
    //starting distrabution on states
    // the sum of the values of all keys should be 1.0
    initial_probabilities: HashMap<S,f64>,
    //A Markov Matrix
    //given states s_1 and s_2 returns the probability of going from s_1 to s_2.
    //For any state s, the sume of values for all keys of the form (s,_) must be 1.0
	transition_matrix: HashMap<(S,S),f64>,
}

impl <S: Eq + Hash + Clone + Debug> MarkovModel<S> {

    //init
    //Purpose:
    //    Creates a Markov Model
    //Pre-conditions:
    //    state_space is non-empty andtransition_matrix has as keys every possible pair of states.
    //    For any state s the sum of all values of all keys of the form (s,_) in transition_matrix is 1.0.
    //    The sum of all values of all keys in initial_probabilities is 1.0.
    pub fn init(
        state_space: HashSet<S>,
        initial_probabilities: HashMap<S,f64>,
        transition_matrix: HashMap<(S,S),f64>,
    ) -> Result<MarkovModel<S>, MMError> {

        //There must be at least one state
        if state_space.len() == 0 {
            return Err(MMError::Syntax("The set of states must be non-empty.".to_string()));
        }

        let mut initial_probabilities_sum = 0.0;
        for state in &state_space {
            if !initial_probabilities.contains_key(&state) {
                return Err(MMError::Syntax("Initial probabilities must contain every state.".to_string()));
            }

            //We know the key exists at this point.
            let initial_state_probability = initial_probabilities.get(&state).unwrap();
            if (initial_state_probability < &0.0) || (initial_state_probability > &1.0) {
                return Err(MMError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
            }
            initial_probabilities_sum += initial_state_probability;


            //check transition matrix keys and probabilities
            let mut total_state_matrix_probability = 0.0;
            for state_2 in &state_space {
                if !transition_matrix.contains_key(&(state.clone(),state_2.clone())) {
                    return Err(MMError::Syntax("Transition matrix must contain every pair of states.".to_string()));
                }

                //we know the key exists at this point.
                let state_to_state_2_prob = transition_matrix.get(&(state.clone(),state_2.clone())).unwrap();
                if (state_to_state_2_prob < &0.0) || (state_to_state_2_prob > &1.0) {
                    return Err(MMError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
                }
                total_state_matrix_probability += state_to_state_2_prob;
            }

            //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
            if (1.0 - total_state_matrix_probability).abs() > FLOAT_ERROR_TOLERANCE {
                return Err(MMError::Syntax("Outgoing probabilites of of a state must sum to 1.".to_string()));
            }
        }

        //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
        if (1.0 - initial_probabilities_sum).abs() > FLOAT_ERROR_TOLERANCE {
            return Err(MMError::Syntax("Total probabilites of the inital probabilites must sum to 1.".to_string()));
        }

        for (state,_) in &initial_probabilities {
            if !state_space.contains(state) {
                    return Err(MMError::Syntax("Every state in initial_probabilites must be a valid state.".to_string()));
                }
        }

        for ((state_1,state_2),_) in &transition_matrix {
            if !state_space.contains(state_1) || !state_space.contains(state_2) {
                    return Err(MMError::Syntax("Every state in transition_matrix must be a valid state.".to_string()));
                }
        }

        Ok(MarkovModel{ state_space, initial_probabilities, transition_matrix, })
    }

    //get_next_state
    //Purpose:
    //    Returns a Option of a state where the state is chosen
    //    with the proabilites given by transition_matrix if
    //    current_state is not None, and initial_probabilities otherwise
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    None
    fn get_next_state(&self, current_state: Option<S>) -> Option<S> {
        let mut threshold = rand::thread_rng().gen_range(0.0, 1.0);
        match current_state {
            //we already have a state
            Some(my_state) => {
                for key in self.state_space.clone() {
                    let prob = *self.transition_matrix.get(&(my_state.clone(),key.clone())).unwrap();
                    if threshold < prob {
                        return Some(key.clone());
                    } else {
                        threshold = threshold - prob;
                    }
                }
                //since the probabilites add up to one, the function should never reach this line
                return None;
            },
            //If we don't have a state we must be starting the sequence.
            None => {
                for key in self.state_space.clone() {
                    let prob = *self.initial_probabilities.get(&key.clone()).unwrap();
                    if threshold < prob {
                        return Some(key.clone());
                    } else {
                        threshold = threshold - prob;
                    }
                }
                //since the probabilites add up to one, the function should never reach this line
                return None;
            }
        }
    }

    //generate_state_sequence
    //Purpose:
    //    Returns a sequence of states of the given length by using the initial_probabilities
    //    and transition_matrix
    //Pre-conditions:
    //    None
    pub fn generate_state_sequence(&self, seq_len: usize) -> Vec<S> {
        let mut generated_sequence: Vec<S> = Vec::new();
        let mut current_state: Option<S> = None;

        for _i in 0..seq_len {
            current_state = self.get_next_state(current_state.clone());
            generated_sequence.push(current_state.clone().unwrap());
        }
        return generated_sequence;
    }

    //prob_of_state_sequence
    //Purpose:
    //    Returns the probability of the Markov Model giving the following sequence of states.
    //Pre-conditions:
    //    state_sequence is non-empty
    pub fn prob_of_state_sequence(&self, state_sequence: Vec<S>) -> f64 {
        //first set the probability to the probability of starting with the first state.
        let mut prob_of_sequence = *self.initial_probabilities.get(&state_sequence[0]).unwrap();
        for i in 1..state_sequence.len() {
            //multiply the probability by the probability of the given transition.
            prob_of_sequence *= *self.transition_matrix.get(&(state_sequence[i-1].clone(),state_sequence[i].clone())).unwrap();
        }
        return prob_of_sequence;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mm_init_errors() {

        //setup
        let s_a = -4;
        let s_b = -5;
        let s_c = -6;
        let s_d = -7;

        let state_space_1: HashSet<isize> = [s_a,s_b,s_c].iter().cloned().collect();
        let state_space_3: HashSet<isize> = [s_a,s_b,s_c,s_d].iter().cloned().collect();


        let initial_probabilities_1: HashMap<isize, f64> =
        [(s_a, 0.4), (s_b, 0.4), (s_c, 0.2),].iter().cloned().collect();
        let initial_probabilities_2: HashMap<isize, f64> =
        [(s_a, 0.7), (s_b, 0.4), (s_c, 0.2),].iter().cloned().collect();
        let initial_probabilities_3: HashMap<isize, f64> =
        [(s_a, -0.2), (s_b, 0.4), (s_c, 0.2),].iter().cloned().collect();

        let transition_matrix_1: HashMap<(isize,isize),f64> =
        [((s_a,s_a), 0.6),((s_a,s_b), 0.3),((s_a,s_c), 0.1),((s_b,s_a), 0.3),((s_b,s_b), 0.6),
        ((s_b,s_c), 0.1),((s_c,s_a), 0.1),((s_c,s_b), 0.1),((s_c,s_c), 0.8),].iter().cloned().collect();
        let transition_matrix_2: HashMap<(isize,isize),f64> =
        [((s_a,s_b), 0.3),((s_a,s_c), 0.1),((s_b,s_a), 0.3),((s_b,s_b), 0.6),
        ((s_b,s_c), 0.1),((s_c,s_a), 0.1),((s_c,s_b), 0.1),((s_c,s_c), 0.8),].iter().cloned().collect();
        let transition_matrix_3: HashMap<(isize,isize),f64> =
        [((s_a,s_a), 0.7),((s_a,s_b), 0.3),((s_a,s_c), 0.1),((s_b,s_a), 0.3),((s_b,s_b), 0.6),
        ((s_b,s_c), 0.1),((s_c,s_a), 0.1),((s_c,s_b), 0.1),((s_c,s_c), 0.8),].iter().cloned().collect();

        //Attempted init with no states
        assert!(MarkovModel::init(HashSet::new(),initial_probabilities_1.clone(),transition_matrix_1.clone()).is_err());

        //Attempted init with not all states accounted for in initial probabilities
        assert!(MarkovModel::init(state_space_3.clone(),initial_probabilities_1.clone(),transition_matrix_1.clone()).is_err());

        //Attempted init with invalid probability in initial probabilites
        assert!(MarkovModel::init(state_space_1.clone(),initial_probabilities_3.clone(),transition_matrix_1.clone()).is_err());

        //Attempted init with sum of initial probabilites exceeding 1.
        assert!(MarkovModel::init(state_space_1.clone(),initial_probabilities_2.clone(),transition_matrix_1.clone()).is_err());

        //Attempted init with transition_matrix not containing every pair.
        assert!(MarkovModel::init(state_space_1.clone(),initial_probabilities_2.clone(),transition_matrix_2.clone()).is_err());

        //Attempted init with transition_matrix containing a row whose probabilites don't sum to 1.
        assert!(MarkovModel::init(state_space_1.clone(),initial_probabilities_2.clone(),transition_matrix_3.clone()).is_err());
    }

}