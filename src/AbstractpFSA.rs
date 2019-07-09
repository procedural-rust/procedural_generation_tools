//Author: Everett Sullivan
//Date Created: 6/16/2019
//Purpose To create an Abstract probabilistic finite state automata (FSA)
//Notes: Another Implementation of pFSA except that the state are a set of
//          objects (which may be different then the alphabet).
//          This was specifally created for ngrams.rs but can be used for other applications.
//Ideas:

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use rand::Rng;

const FLOAT_ERROR_TOLERANCE: f64 = 0.001;

////////////////////
//Custom Error handling code
////////////////////

#[derive(Debug)]
pub enum APFSAError {
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for APFSAError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            APFSAError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for APFSAError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            APFSAError::Syntax(ref _err_string)  => None,
        }
    }
}

////////////////////
//Abstract pFSA code
////////////////////

#[derive(Debug)]
pub struct AbstractProbabilisticFiniteStateAutomata<S: Eq + Hash + Clone, T: Eq + Hash + Clone> {
    states: HashSet<S>,
    alphabet: HashSet<T>,
	transitions: HashMap<(S,T),(S,f64)>,
	start_state: S,
    accepting_states: HashSet<S>
}

impl <S: Eq + Hash + Clone, T: Eq + Hash + Clone> AbstractProbabilisticFiniteStateAutomata<S,T> {

    //init
    //Purpose:
    //    Creates a Abstract pFSA.
    //Pre-conditions:
    //    states is non-empty, alphabet is non-empty, transitions has as keys every possible state letter pair.
    //    The total probability of all keys (s,_) for a state s must add up to 1 unless s is a accepting_state
    //    in which they must add to less than 1.
    pub fn init(
        states: HashSet<S>,
        alphabet: HashSet<T>,
        transitions: HashMap<(S,T),(S,f64)>,
        start_state: S,
        accepting_states: HashSet<S>
    ) -> Result<AbstractProbabilisticFiniteStateAutomata<S,T>, APFSAError> {

        //There must be at least one state.
        if states.len() == 0 {
            return Err(APFSAError::Syntax("States must be non-empty.".to_string()));
        }

        //There must be at least one letter in the alphabet
        if alphabet.len() == 0 {
            return Err(APFSAError::Syntax("Alphabet must be non-empty.".to_string()));
        }

        //Every (state,letter) pair must be a key in Transitions.
        for (state, letter) in states.iter().zip(alphabet.iter()) {
            if !transitions.contains_key(&(state.clone(), letter.clone())) {
                return Err(APFSAError::Syntax("Transitions must contain every (state,letter) pair.".to_string()));
            }
        }

        //Every transition in Transitions must be a valid (state,letter) pair.
        for ((state, letter), (next_state, prob)) in &transitions {
            if !states.contains(&state) {
                return Err(APFSAError::Syntax("Transitions must be from valid states.".to_string()));
            } else if !alphabet.contains(&letter) {
                return Err(APFSAError::Syntax("Transitions must have valid letters.".to_string()));
            } else if !states.contains(&next_state) {
                return Err(APFSAError::Syntax("Transitions must go to valid states.".to_string()));
            } else if (prob < &0.0) || (prob > &1.0) {
                return Err(APFSAError::Syntax("Transitions probability must be a valid probability (between 0 and 1 inclusive).".to_string()));
            }
        }

        //The starting state must be a valid state
        if !states.contains(&start_state) {
            return Err(APFSAError::Syntax("start_state must be a valid state.".to_string()));
        }

        //All accepting state must be valid states
        for accepting_state in &accepting_states {
            if !states.contains(&accepting_state) {
                return Err(APFSAError::Syntax("Accepting states must be valid states.".to_string()));
            }
        }

        for state in &states {
            let mut total_prob: f64  = 0.0;
            for letter in &alphabet {
                //since we have checked every (state,letter) pair we know this keys exists, then we grab the probability associated with it.
                total_prob += transitions.get(&(state.clone(),letter.clone())).unwrap().1;
            }

            if accepting_states.contains(&state) {
                //since we know that probabilites are positives we only need to check that they sum to less than one.
                if total_prob >= 1.0 {
                    return Err(APFSAError::Syntax("Outgoing probabilites of an accepting state must sum to less than 1.".to_string()));
                }
            }else {
                //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
                if (1.0 - total_prob).abs() > FLOAT_ERROR_TOLERANCE {
                    return Err(APFSAError::Syntax("Outgoing probabilites of an non-accepting state must sum to 1.".to_string()));
                }
            }
        }

        Ok(AbstractProbabilisticFiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, })
    }

    //unsafe_init
    //Purpose:
    //    Creates a Abstract pFSA.
    //Pre-conditions:
    //    states is non-empty, alphabet is non-empty, transitions has as keys every possible state letter pair.
    //    The total probability of all keys (s,_) for a state s must add up to 1 unless s is a accepting_state
    //    in which they must add to less than 1.
    //    Does not check that the conditions hold.
    pub fn unsafe_init(
        states: HashSet<S>,
        alphabet: HashSet<T>,
        transitions: HashMap<(S,T),(S,f64)>,
        start_state: S,
        accepting_states: HashSet<S>
    ) -> AbstractProbabilisticFiniteStateAutomata<S,T> {
        AbstractProbabilisticFiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, }
    }

    //getters
    pub fn get_states(&self) -> HashSet<S> {
        self.states.clone()
    }

    pub fn get_alphabet(&self) -> HashSet<T> {
        self.alphabet.clone()
    }

    pub fn get_start_state(&self) -> S {
        self.start_state.clone()
    }

    pub fn get_accepting_states(&self) -> HashSet<S> {
        self.accepting_states.clone()
    }

    //step
    //Purpose:
    //    Given a state and a letter, returns a tuple with the resulting state and the probability
    //    of that letter being chosen at the current state.
    //Pre-conditions:
    //    current_state must be a valid state and letter must be in alphabet.
    pub fn step(&self, current_state: S, letter: T) -> Option<&(S, f64)> {
        self.transitions.get(&(current_state,letter))
    }

    //get_next_step
    //Purpose:
    //    Returns a Option of a tuple containing a letter and state where the letter is chosen
    //    with the proabilites given by transitions and the state is the state the machine will be in next.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    current_state must be a valid state.
    pub fn get_next_step(&self, current_state: S) -> Option<(T, S)> {
        let mut threshold = rand::thread_rng().gen_range(0.0, 1.0);
        for key in self.transitions.keys() {
            let (a,b) = key.clone();
            if a == current_state {
                let (c,d) = self.transitions.get(key).unwrap().clone();
                //reduce threshold if not picked.
                if threshold < d {
                    return Some((b,c));
                } else {
                    threshold = threshold - d;
                }
            }
        }
        //Will only return if key doesn't exist (An error) or if the path ended (in which the current
        //  state must be an ending state).
        return None;
    }

    //generate
    //Purpose:
    //    Generates a word obtained and moving through the Abstract pFSA.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    None
    pub fn generate(&self) -> Vec<T> {
        let mut current_state = self.start_state.clone();
        let mut word = Vec::new();
        loop {
            match self.get_next_step(current_state) {
                Some((letter, state)) => {
                    //add output to end of word, move to next state.
                    word.push(letter);
                    current_state = state;
                },
                None => {
                    //reached an ending state.
                    return word;
                },
            }
        }
    }

    //prob_of_word
    //Purpose:
    //    Returns the probability of a given word being the result of the output of the Abstract pFSA.
    //    Since floats are used the probability may not be exact.
    //Pre-conditions:
    //    All letters in word must be in the alphabet
    pub fn prob_of_word(&self, word: &Vec<T>) -> Result<f64, APFSAError> {
        let mut current_state = self.start_state.clone(); //Need to track current state
        let mut prob_of_word = 1.0;
        for letter in word {
            if !self.alphabet.contains(&letter) {
                return Err(APFSAError::Syntax("Word contains an invalid letter.".to_string()));
            }
            let (next_state,prob_of_letter) = self.transitions.get(&(current_state,letter.clone())).unwrap();
            current_state = next_state.clone();
            prob_of_word = prob_of_word*prob_of_letter;
        }

        //compute probability of ending
        let mut prob_of_ending = 1.0;
        for key in self.transitions.keys() {
            let (a,_) = key.clone();
            if a == current_state {
                let (_,d) = self.transitions.get(key).unwrap().clone();
                prob_of_ending = prob_of_ending - d;
            }
        }

        Ok(prob_of_word*prob_of_ending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstract_pfsa_init_errors() {

        //setup
        let a = -1;
        let b = -2;

        let states_1: HashSet<usize> = [1,2].iter().cloned().collect();
        let states_2: HashSet<usize> = [1].iter().cloned().collect();
        let alphabet_1: HashSet<isize> =  [a].iter().cloned().collect();
        let alphabet_2: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states_1: HashSet<usize> = [1].iter().cloned().collect();
        let accepting_states_2: HashSet<usize> = [0].iter().cloned().collect();
        let accepting_states_4: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans_1: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1,0.3)),((1,b), (2, 0.3)),((2,a), (1,0.5)),((2,b), (1,0.5)),].iter().cloned().collect();
        let trans_2: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1,0.3)),((1,b), (2,0.3)),].iter().cloned().collect();
        let trans_3: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1, 0.3)),((1,b), (2, 0.3)),].iter().cloned().collect();
        let trans_4: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1, 0.3)),((1,b), (1, -0.1)),].iter().cloned().collect();
        let trans_5: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1, 0.3)),((1,b), (1, 1.2)),].iter().cloned().collect();
        let trans_6: HashMap<(usize, isize), (usize, f64)> =
            [((1,a), (1,0.3)),((1,b), (2, 0.3)),((2,a), (1,0.3)),((2,b), (1,0.3)),].iter().cloned().collect();

        //Attempted init with empty states (states must be non-empty)
        assert!(AbstractProbabilisticFiniteStateAutomata::init(HashSet::new(),alphabet_2.clone(),trans_1.clone(),1,accepting_states_1.clone()).is_err());

        //Attempted init with empty alphabet (alphabet must be a non-empty)
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_1.clone(),HashSet::new(),trans_1.clone(),1,accepting_states_1.clone()).is_err());

        //Attempted init with invalid start_state
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_1.clone(),alphabet_2.clone(),trans_1.clone(),0,accepting_states_1.clone()).is_err());

        //Attempted init with invalid as an accepting state
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_1.clone(),alphabet_2.clone(),trans_1.clone(),2,accepting_states_2.clone()).is_err());

        //Attempted init with a transition matrix with letter not used in alphabet.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_1.clone(),alphabet_1.clone(),trans_1.clone(),2,accepting_states_1.clone()).is_err());

        //Attempted init with a transition matrix not containing all (state,letter) pairs.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_1.clone(),alphabet_2.clone(),trans_2.clone(),2,accepting_states_1.clone()).is_err());

        //Attempted init with a transition matrix containing a transition to a invalid state
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_2.clone(),alphabet_2.clone(),trans_3.clone(),1,accepting_states_1.clone()).is_err());

        //Attempted init with a transition matrix containing a probability less than 0.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_2.clone(),alphabet_2.clone(),trans_4.clone(),1,accepting_states_1.clone()).is_err());

        //Attempted init with a transition matrix containing a probability greater than 1.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_2.clone(),alphabet_2.clone(),trans_5.clone(),1,accepting_states_1.clone()).is_err());

        //Attempted init with a transition matrix containing an accepting state for which its outgoing probabilites sum to 1.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_2.clone(),alphabet_2.clone(),trans_1.clone(),1,accepting_states_4.clone()).is_err());

        //Attempted init with a transition matrix containing a non-accepting state for which its outgoring porbabilites don't sum to 1.
        assert!(AbstractProbabilisticFiniteStateAutomata::init(states_2.clone(),alphabet_2.clone(),trans_6.clone(),1,accepting_states_1.clone()).is_err());
    }

    #[test]
    fn abstract_pfsa_getters() {
        //setup
        let a = -1;
        let b = -2;

        let state_space: HashSet<usize> =  [1,2,3].iter().cloned().collect();
        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), (usize,f64)> =
            [((1,a), (1,0.5)), ((1,b), (2,0.4)), ((2,a), (1,0.5)), ((2,b), (3,0.3)), ((3,a), (1,0.5)), ((3,b), (3,0.5)),].iter().cloned().collect();

        let test_apfsa = AbstractProbabilisticFiniteStateAutomata::init(state_space.clone(),alphabet.clone(),trans,1,accepting_states.clone()).unwrap();

        assert_eq!(test_apfsa.get_states(),state_space);

        assert_eq!(test_apfsa.get_alphabet(),alphabet);

        assert_eq!(test_apfsa.get_start_state(),1);

        assert_eq!(test_apfsa.get_accepting_states(),accepting_states);
    }

    #[test]
    fn abstract_pfsa_transitions() {
        //setup
        let a = -1;
        let b = -2;

        let state_space: HashSet<usize> =  [1,2,3].iter().cloned().collect();
        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), (usize,f64)> =
            [((1,a), (1,0.5)), ((1,b), (2,0.4)), ((2,a), (1,0.5)), ((2,b), (3,0.3)), ((3,a), (1,0.5)), ((3,b), (3,0.5)),].iter().cloned().collect();

        let test_apfsa = AbstractProbabilisticFiniteStateAutomata::init(state_space.clone(),alphabet.clone(),trans.clone(),1,accepting_states.clone()).unwrap();

        for ((state, letter), (next_state, prob)) in &trans {
            assert_eq!(test_apfsa.step(*state,*letter).unwrap(),&(*next_state, prob.clone()));
        }
    }

    #[test]
    fn abstract_pfsa_generate() {

        //setup
        let a = -1;
        let b = -2;

        let state_space: HashSet<usize> =  [1,2].iter().cloned().collect();
        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let final_states: HashSet<usize> = [2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), usize> =
        [((1,a), 1),
        ((1,b), 2),
        ((2,a), 1),
        ((2,b), 2),]
        .iter().cloned().collect();
        let trans_prob: HashMap<(usize, isize), (usize, f64)> =
        [((1,a), (1,0.5)),
        ((1,b), (2,0.5)),
        ((2,a), (1,0.3)),
        ((2,b), (2,0.3)),]
        .iter().cloned().collect();
        let my_fsa = crate::fsa::FiniteStateAutomata::init((2,alphabet.clone(),trans,1,final_states.clone())).unwrap();
        let my_apfsa = AbstractProbabilisticFiniteStateAutomata::init(state_space.clone(),alphabet.clone(),trans_prob,1,final_states.clone()).unwrap();

        for _i in 0..1000 {
            let generated_sequence = my_apfsa.generate();
            assert!(my_fsa.accepts_sequence(&generated_sequence).unwrap());
        }
    }

    #[test]
    fn abstract_pfsa_prob_of_word() {

        //setup
        let a = -1;
        let b = -2;

        let state_space: HashSet<usize> =  [1,2].iter().cloned().collect();
        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let final_states: HashSet<usize> = [1].iter().cloned().collect();
        let trans_prob: HashMap<(usize, isize), (usize, f64)> =
        [((1,a), (1,0.3)),
        ((1,b), (2,0.3)),
        ((2,a), (1,0.5)),
        ((2,b), (2,0.5)),]
        .iter().cloned().collect();
        let my_apfsa = AbstractProbabilisticFiniteStateAutomata::init(state_space.clone(),alphabet.clone(),trans_prob,1,final_states.clone()).unwrap();

        assert!( (my_apfsa.prob_of_word(&vec![]).unwrap() - 0.4).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![a]).unwrap() - 0.12).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![b,a]).unwrap() - 0.06).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![b,b,a]).unwrap() - 0.03).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![a,b,a]).unwrap() - 0.018).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![b,b,b,a]).unwrap() - 0.015).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![b]).unwrap() - 0.0).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_apfsa.prob_of_word(&vec![a,a,a,b]).unwrap() - 0.0).abs() < FLOAT_ERROR_TOLERANCE );
    }

}