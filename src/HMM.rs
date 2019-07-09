//Author: Everett Sullivan
//Date Created: 6/20/2019
//Purpose To create a Hidden Markov Model.
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
pub enum HMMError {
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for HMMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            HMMError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for HMMError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            HMMError::Syntax(ref _err_string)  => None,
        }
    }
}

////////////////////
//Hidden Markov Model code
////////////////////

#[derive(Debug)]
pub struct HiddenMarkovModel<S: Eq + Hash + Clone + Debug, T: Eq + Hash + Clone + Debug> {
    state_space: HashSet<S>,
    observation_space: HashSet<T>,
    //starting distrabution on states
    // the sum of the values of all keys should be 1.0
    initial_probabilities: HashMap<S,f64>,
    //A Markov Matrix
    //given states s_1 and s_2 returns the probability of going from s_1 to s_2.
    //For any state s, the sume of values for all keys of the form (s,_) must be 1.0
	transition_matrix: HashMap<(S,S),f64>,
    //given state s and observation o returns the probability of s emitting o.
    //For any state s, the sume of values for all keys of the form (s,_) must be 1.0
	emission_matrix: HashMap<(S,T),f64>,
}

impl <S: Eq + Hash + Clone + Debug, T: Eq + Hash + Clone + Debug> HiddenMarkovModel<S,T> {

    //init
    //Purpose:
    //    Creates a Hidden Markov Model
    //Pre-conditions:
    //    state_space is non-empty, observation_space is non-empty, transition_matrix has as keys every possible pair of states.
    //    emission_matrix has as keys every possible pair of state and emission.
    //    For any state s the sum of all values of all keys of the form (s,_) in transition_matrix is 1.0.
    //    For any state s the sum of all values of all keys of the form (s,_) in emission_matrix is 1.0.
    //    The sum of all values of all keys in initial_probabilities is 1.0.
    pub fn init(
        state_space: HashSet<S>,
        observation_space: HashSet<T>,
        initial_probabilities: HashMap<S,f64>,
        transition_matrix: HashMap<(S,S),f64>,
        emission_matrix: HashMap<(S,T),f64>,
    ) -> Result<HiddenMarkovModel<S,T>, HMMError> {

        //There must be at least one state
        if state_space.len() == 0 {
            return Err(HMMError::Syntax("The set of states must be non-empty.".to_string()));
        }

        //There must be at least one observation
        if observation_space.len() == 0 {
            return Err(HMMError::Syntax("The set of observations must be non-empty.".to_string()));
        }

        let mut initial_probabilities_sum = 0.0;
        for state in &state_space {
            if !initial_probabilities.contains_key(&state) {
                return Err(HMMError::Syntax("Initial probabilities must contain every state.".to_string()));
            }

            //We know the key exists at this point.
            let initial_state_probability = initial_probabilities.get(&state).unwrap();
            if (initial_state_probability < &0.0) || (initial_state_probability > &1.0) {
                return Err(HMMError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
            }
            initial_probabilities_sum += initial_state_probability;


            //check transition matrix keys and probabilities
            let mut total_state_matrix_probability = 0.0;
            for state_2 in &state_space {
                if !transition_matrix.contains_key(&(state.clone(),state_2.clone())) {
                    return Err(HMMError::Syntax("Transition matrix must contain every pair of states.".to_string()));
                }

                //we know the key exists at this point.
                let state_to_state_2_prob = transition_matrix.get(&(state.clone(),state_2.clone())).unwrap();
                if (state_to_state_2_prob < &0.0) || (state_to_state_2_prob > &1.0) {
                    return Err(HMMError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
                }
                total_state_matrix_probability += state_to_state_2_prob;
            }

            //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
            if (1.0 - total_state_matrix_probability).abs() > FLOAT_ERROR_TOLERANCE {
                return Err(HMMError::Syntax("Outgoing probabilites of of a state must sum to 1.".to_string()));
            }


            //check emission matrix keys and probabilities
            let mut total_state_emission_probability = 0.0;
            for observation in &observation_space {
                if !emission_matrix.contains_key(&(state.clone(),observation.clone())) {
                    return Err(HMMError::Syntax("Emission matrix must contain every state observation pair.".to_string()));
                }

                //we know the key exists at this point.
                let emission_at_state_prob = emission_matrix.get(&(state.clone(),observation.clone())).unwrap();
                if (emission_at_state_prob < &0.0) || (emission_at_state_prob > &1.0) {
                    return Err(HMMError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
                }
                total_state_emission_probability += emission_at_state_prob;
            }

            //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
            if (1.0 - total_state_emission_probability).abs() > FLOAT_ERROR_TOLERANCE {
                return Err(HMMError::Syntax("Emission probabilites of each state must sum to 1.".to_string()));
            }
        }

        //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
        if (1.0 - initial_probabilities_sum).abs() > FLOAT_ERROR_TOLERANCE {
            return Err(HMMError::Syntax("Total probabilites of the inital probabilites must sum to 1.".to_string()));
        }

        for (state,_) in &initial_probabilities {
            if !state_space.contains(state) {
                    return Err(HMMError::Syntax("Every state in initial_probabilites must be a valid state.".to_string()));
                }
        }

        for ((state_1,state_2),_) in &transition_matrix {
            if !state_space.contains(state_1) || !state_space.contains(state_2) {
                    return Err(HMMError::Syntax("Every state in transition_matrix must be a valid state.".to_string()));
                }
        }

        for ((state,observation),_) in &emission_matrix {
            if !state_space.contains(state) || !observation_space.contains(observation) {
                    return Err(HMMError::Syntax("Every state and observation in emission_matrix must be a valid state or observation.".to_string()));
                }
        }

        Ok(HiddenMarkovModel{ state_space, observation_space, initial_probabilities, transition_matrix, emission_matrix, })
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

    //get_emission_at_state
    //Purpose:
    //    Returns an emission from state current_state where the emission is chosen
    //    with the proabilites given by emission_matrix.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //     current_state is in state_space
    fn get_emission_at_state(&self, current_state: S) -> T {
        let mut threshold = rand::thread_rng().gen_range(0.0, 1.0);
        for key in self.observation_space.clone() {
            let prob = *self.emission_matrix.get(&(current_state.clone(),key.clone())).unwrap();
            if threshold < prob {
                return key.clone();
            } else {
                threshold = threshold - prob;
            }
        }
        //since the probabilites add up to one, the function should never reach this line
        panic!("Probabilities of emission at some state don't add up to 1.0");
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

    //generate_emission_sequence
    //Purpose:
    //    Returns a sequence of emissions of the given length by using initial_probabilities,
    //    transition_matrix, and emission_matrix
    //Pre-conditions:
    //    None
    pub fn generate_emission_sequence(&self, seq_len: usize) -> Vec<T> {
        let mut generated_sequence: Vec<T> = Vec::new();
        let mut current_state: Option<S> = None;

        for _i in 0..seq_len {
            current_state = self.get_next_state(current_state.clone());
            let my_emssion = self.get_emission_at_state(current_state.clone().unwrap());
            generated_sequence.push(my_emssion.clone());
        }
        return generated_sequence;
    }

    //generate_state_emission_sequence
    //Purpose:
    //    Returns a sequence of states with emissions of the given length by using initial_probabilities,
    //    transition_matrix, and emission_matrix.
    //Pre-conditions:
    //    None
    pub fn generate_state_emission_sequence(&self, seq_len: usize) -> Vec<(S,T)> {
        let mut generated_sequence: Vec<(S,T)> = Vec::new();
        let mut current_state: Option<S> = None;

        for _i in 0..seq_len {
            current_state = self.get_next_state(current_state.clone());
            let my_emssion = self.get_emission_at_state(current_state.clone().unwrap());
            generated_sequence.push((current_state.clone().unwrap(),my_emssion.clone()));
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

    //prob_of_observation_sequence
    //Purpose:
    //    Returns the probability of the Hidden Markov Model giving the following sequence of observations.
    //Pre-conditions:
    //    observation_sequence is non-empty
    pub fn prob_of_observation_sequence(&self, observation_sequence: Vec<T>) -> f64 {

        //create a table that will keep track of probabilites.
        //the key (my_state,index) will return the probability that the observation sequence
        //up to index has been observed and that we ended in state my_state.
        let mut probability_chart: HashMap<(S,usize),f64> = HashMap::new();

        //initialize with emissions from the start state.
        for state in &self.state_space {
            let prob_of_starting_in_state = *self.initial_probabilities.get(&state).unwrap();
            let prob_of_emitting_from_state = *self.emission_matrix.get(&(state.clone(),observation_sequence[0].clone())).unwrap();
            probability_chart.insert((state.clone(), 0), prob_of_starting_in_state*prob_of_emitting_from_state);
        }

        for i in 1..observation_sequence.len() {
            for state in &self.state_space {
                //initialize key so that accessing it will always have something.
                probability_chart.insert((state.clone(), i), 0.0);
                //sum over all possible paths to the current state by looking at the pervious column
                for previous_state in &self.state_space {
                    let prob_of_previous_state = probability_chart.get(&(previous_state.clone(),i-1)).unwrap().clone();
                    let prob_of_transition = self.transition_matrix.get(&(previous_state.clone(),state.clone())).unwrap();
                    let prob_of_emission = self.emission_matrix.get(&(state.clone(),observation_sequence[i].clone())).unwrap();
                    let (_,prob) = probability_chart.remove_entry(&(state.clone(),i)).unwrap();
                    probability_chart.insert((state.clone(),i),prob + prob_of_previous_state*prob_of_transition*prob_of_emission);
                }
            }
        }

        //sinice we don't care which state we ended at, sum the porbabilites in the last column to get the desired result.
        let mut prob_of_observation_seq = 0.0;
        for state in &self.state_space {
            prob_of_observation_seq += probability_chart.get(&(state.clone(),observation_sequence.len()-1)).unwrap();
        }
        return prob_of_observation_seq;
    }

    //prob_of_state_emission_sequence
    //Purpose:
    //    Returns the probability of the Markov Model giving the following sequence of states.
    //Pre-conditions:
    //    state_sequence is non-empty
    pub fn prob_of_state_emission_sequence(&self, state_sequence: Vec<(S,T)>) -> f64 {
        //first set the probability to the probability of starting with the first state.
        let mut prob_of_sequence = *self.initial_probabilities.get(&state_sequence[0].0).unwrap();
        for i in 1..state_sequence.len() {
            //multiply the probability by the probability of the given transition.
            prob_of_sequence *= *self.transition_matrix.get(&(state_sequence[i-1].clone().0,state_sequence[i].clone().0)).unwrap();
            //multiply the probability by the probability of the given emission.
            prob_of_sequence *= *self.emission_matrix.get(&(state_sequence[i-1].clone().0,state_sequence[i].clone().1)).unwrap();
        }
        return prob_of_sequence;
    }

    //viterbi
    //Purpose:
    //    Given a sequence of observations, returns the sequence of states most likely to have procuded such observations
    //    and the probability that this specific observation-state pair happening.
    //Pre-conditions:
    //    state_sequence is non-empty
    pub fn viterbi(&self, observation_sequence: Vec<T>) -> (Vec<S>,f64) {
        //create a table that will keep track of probabilites.
        //the key (my_state,index) will return the probability that the observation sequence
        //up to index has been observed and that we ended in state my_state.
        let mut probability_chart: HashMap<(S,usize),(Vec<S>,f64)> = HashMap::new();

        //initialize with emissions from the start state.
        for state in &self.state_space {
            let prob_of_starting_in_state = *self.initial_probabilities.get(&state).unwrap();
            let prob_of_emitting_from_state = *self.emission_matrix.get(&(state.clone(),observation_sequence[0].clone())).unwrap();
            probability_chart.insert((state.clone(), 0), (vec![state.clone()],prob_of_starting_in_state*prob_of_emitting_from_state));
        }

        for i in 1..observation_sequence.len() {
            for state in &self.state_space {
                //initialize key so that accessing it will always have something.
                probability_chart.insert((state.clone(), i), (Vec::new(),0.0));
                //find the entry in the pervious column that has the highest proability of continuting to the current state
                for previous_state in &self.state_space {
                    let prob_of_current_state = probability_chart.get(&(state.clone(),i)).unwrap().1.clone();
                    let prob_of_previous_state = probability_chart.get(&(previous_state.clone(),i-1)).unwrap().1.clone();
                    let prob_of_transition = self.transition_matrix.get(&(previous_state.clone(),state.clone())).unwrap();
                    let prob_of_emission = self.emission_matrix.get(&(state.clone(),observation_sequence[i].clone())).unwrap();
                    //if we found one, update the data.
                    if prob_of_current_state < prob_of_previous_state*prob_of_transition*prob_of_emission {
                        //push the current state onto the path vector.
                        let mut vec_up_to = probability_chart.get(&(previous_state.clone(),i-1)).unwrap().0.clone();
                        vec_up_to.push(state.clone());
                        probability_chart.remove_entry(&(state.clone(),i)).unwrap();
                        probability_chart.insert((state.clone(),i),(vec_up_to,prob_of_previous_state*prob_of_transition*prob_of_emission));
                    }
                }
            }
        }

        //find a state in the last column with highest probability return that entry.
        let mut max_porb_seq_pair = (Vec::new(),0.0);
        for state in &self.state_space {
            if probability_chart.get(&(state.clone(),observation_sequence.len()-1)).unwrap().1 > max_porb_seq_pair.1 {
                max_porb_seq_pair = probability_chart.get(&(state.clone(),observation_sequence.len()-1)).unwrap().clone();
            }
        }
        return max_porb_seq_pair;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmm_init_errors() {

        //setup
        let o_a = -1;
        let o_b = -2;
        let o_c = -3;
        let s_a = -4;
        let s_b = -5;
        let s_c = -6;
        let s_d = -7;

        let state_space_1: HashSet<isize> = [s_a,s_b,s_c].iter().cloned().collect();
        let state_space_3: HashSet<isize> = [s_a,s_b,s_c,s_d].iter().cloned().collect();

        let observation_space_1: HashSet<isize> = [o_a,o_b,o_c].iter().cloned().collect();


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

        let emission_matrix_1: HashMap<(isize,isize),f64> =
        [((s_a,o_a), 0.7),((s_a,o_b), 0.1),((s_a,o_c), 0.2), ((s_b,o_a), 0.1),((s_b,o_b), 0.7),
        ((s_b,o_c), 0.2),((s_c,o_a), 0.33),((s_c,o_b), 0.33),((s_c,o_c), 0.34),].iter().cloned().collect();
        let emission_matrix_2: HashMap<(isize,isize),f64> =
        [((s_a,o_b), 0.1),((s_a,o_c), 0.2), ((s_b,o_a), 0.1),((s_b,o_b), 0.7),
        ((s_b,o_c), 0.2),((s_c,o_a), 0.33),((s_c,o_b), 0.33),((s_c,o_c), 0.34),].iter().cloned().collect();
        let emission_matrix_3: HashMap<(isize,isize),f64> =
        [((s_a,o_a), 0.8),((s_a,o_b), 0.1),((s_a,o_c), 0.2), ((s_b,o_a), 0.1),((s_b,o_b), 0.7),
        ((s_b,o_c), 0.2),((s_c,o_a), 0.33),((s_c,o_b), 0.33),((s_c,o_c), 0.34),].iter().cloned().collect();

        //Attempted init with no states
        assert!(HiddenMarkovModel::init(HashSet::new(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_1.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with no observations
        assert!(HiddenMarkovModel::init(state_space_1.clone(),HashSet::new(),initial_probabilities_1.clone(),transition_matrix_1.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with not all states accounted for in initial probabilities
        assert!(HiddenMarkovModel::init(state_space_3.clone(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_1.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with invalid probability in initial probabilites
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_3.clone(),transition_matrix_1.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with sum of initial probabilites exceeding 1.
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_2.clone(),transition_matrix_1.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with transition_matrix not containing every pair.
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_2.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with transition_matrix containing a row whose probabilites don't sum to 1.
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_3.clone(),emission_matrix_1.clone()).is_err());

        //Attempted init with emission_matrix not containing every pair.
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_1.clone(),emission_matrix_2.clone()).is_err());

        //Attempted init with emission_matrix containing a row whose probabilites don't sum to 1.
        assert!(HiddenMarkovModel::init(state_space_1.clone(),observation_space_1.clone(),initial_probabilities_1.clone(),transition_matrix_1.clone(),emission_matrix_3.clone()).is_err());
    }

}
