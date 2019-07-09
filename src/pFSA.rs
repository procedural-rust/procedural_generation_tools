//Author: Everett Sullivan
//Date Created: 6/16/2019
//Purpose To create a probabilistic finite state automata (pFSA)
//Notes:
//Ideas: the function step and get_next_step returns options which
//          already take care of all input errors.
//          Should it be left like this or have them return Results
//          with a PFSAError like in FSA.rs?

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;
use rand::Rng;

const FLOAT_ERROR_TOLERANCE: f64 = 0.001;

////////////////////
//Custom Error handling code
////////////////////

#[derive(Debug)]
pub enum PFSAError {
    Io(io::Error),
    ParseInt(num::ParseIntError),
    ParseFloat(num::ParseFloatError),
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for PFSAError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PFSAError::Io(ref err) => err.fmt(f),
            PFSAError::ParseInt(ref err) => err.fmt(f),
            PFSAError::ParseFloat(ref err) => err.fmt(f),
            PFSAError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for PFSAError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            PFSAError::Io(ref err) => Some(err),
            PFSAError::ParseInt(ref err) => Some(err),
            PFSAError::ParseFloat(ref err) => Some(err),
            PFSAError::Syntax(ref _err_string)  => None,
        }
    }
}

use std::io;
use std::num;

// need to turn errors into the custum type, used by the ? operator to convert errors.
impl From<io::Error> for PFSAError {
    fn from(err: io::Error) -> PFSAError {
        PFSAError::Io(err)
    }
}

impl From<num::ParseIntError> for PFSAError {
    fn from(err: num::ParseIntError) -> PFSAError {
        PFSAError::ParseInt(err)
    }
}

impl From<num::ParseFloatError> for PFSAError {
    fn from(err: num::ParseFloatError) -> PFSAError {
        PFSAError::ParseFloat(err)
    }
}

////////////////////
//pFSA code
////////////////////

pub struct ProbabilisticFiniteStateAutomata<T: Eq + Hash + Clone> {
    //Since states are enumerated by positive integers it suffice
    //to return the number of states to know what all the states are.
    states: usize,
    alphabet: HashSet<T>,
	transitions: HashMap<(usize,T),(usize,f64)>,
	start_state: usize,
    accepting_states: HashSet<usize>
}

impl ProbabilisticFiniteStateAutomata<String> {

    //data_from_file
    //Purpose:
    //    Creates the data needed to construct a pFSA from a file.
    //Pre-conditions:
    //    file is in the correct format.
    //Note:
    //  file are arranged as follows:
    //  The first line contains the number of states (must be a positive integer).
    //  The second line contains the alphabet of the pFSA where each letter is
    //      sperated by whitespace (must have at least one letter).
    //      (Note that "letter" here just means a element of the alphabet, the "letter"
    //      can be anything abstractly but if wer are reading from a file we assume they are strings.)
    //  The thrid line has the (single) start state
    //  The fourth line contains the accepting states seperated by whitespace
    //  All following lines are the transition instructions.
    //      Each line contains, in order, the state it is starting from, the
    //      letter (or label) of the transition, and the state it will go to with
    //      that input, and the probability that step will be taken (a f64 between 0 and 1).
    //      Note that every pair of state and letter must have a transition.
    //      Also for any state all the rules which start with it must have their probabilites add
    //      up 1, unless the state is an accepting state in which they must add up to lesss than one.
    pub fn data_from_file(filename: String) -> Result<(usize, HashSet<String>, HashMap<(usize,String),(usize,f64)>, usize, HashSet<usize>), PFSAError> {
        // Open file.
        let contents = fs::read_to_string(filename)?;

        // Break file into lines.
        let lines: Vec<&str> = contents.lines().collect();

        //A seperate line is needed for number of state, the alphabet, starting state, accepting states, and each transition.
        if lines.len() < 5 {
            return Err(PFSAError::Syntax("A pFSA file requires at least five lines.".to_string()));
        }

        // get number of states
        let states: usize = lines[0].parse()?;

        // get alphabet
        let mut alphabet = HashSet::new();
        let alphabet_string = lines[1].split_whitespace().collect::<Vec<&str>>();
        for letter in alphabet_string {
            alphabet.insert(letter.to_string());
        }

        //get starting state
        let start_state: usize = lines[2].parse()?;

        //get accepting states
        let mut accepting_states = HashSet::new();
        let accepting_states_string = lines[3].split_whitespace().collect::<Vec<&str>>();
        for state in accepting_states_string {
            accepting_states.insert(state.parse::<usize>()?);
        }

        //get transitions
        let mut transitions = HashMap::new();
        for i in 4..lines.len() {
            let transition_string = lines[i].split_whitespace().collect::<Vec<&str>>();
            if transition_string.len() != 4 {
                return Err(PFSAError::Syntax("A transition requires four parts.".to_string()));
            }

            let current_state = transition_string[0].parse::<usize>()?;

            let current_letter = transition_string[1].to_string();

            let next_state = transition_string[2].parse::<usize>()?;

            let probability = transition_string[3].parse::<f64>()?;

            transitions.insert((current_state,current_letter),(next_state,probability));
        }

        Ok((states, alphabet, transitions, start_state, accepting_states))
    }

}

impl <T: Eq + Hash + Clone> ProbabilisticFiniteStateAutomata<T> {

    //init
    //Purpose:
    //    Creates a pFSA.
    //Pre-conditions:
    //    states is positive, alphabet is non-empty, transitions has as keys every possivle state letter pair.
    //    The total probability of all keys (s,_) for a state s must add up to 1 unless s is a accepting_state
    //    in which they must add to less than 1.
    pub fn init(
        (states, alphabet, transitions, start_state, accepting_states): (usize, HashSet<T>, HashMap<(usize,T),(usize,f64)>, usize, HashSet<usize>,)
    ) -> Result<ProbabilisticFiniteStateAutomata<T>, PFSAError> {

        //There must be at least one state.
        if states <= 0 {
            return Err(PFSAError::Syntax("States must be a positive integer.".to_string()));
        }

        //There must be at least one letter in the alphabet
        if alphabet.len() == 0 {
            return Err(PFSAError::Syntax("Alphabet must be non-empty.".to_string()));
        }

        //Every (state,letter) pair must be a key in Transitions.
        let numbers = 1..;
        let state_iter = numbers.take(states);
        for (state, letter) in state_iter.zip(alphabet.iter()) {
            if !transitions.contains_key(&(state, letter.clone())) {
                return Err(PFSAError::Syntax("Transitions must contain every (state,letter) pair.".to_string()));
            }
        }

        //Every transition in Transitions must be a valid (state,letter) pair.
        for ((state, letter), (next_state, prob)) in &transitions {
            if (state <= &0) || (state > &states) {
                return Err(PFSAError::Syntax("Transitions must be from valid states.".to_string()));
            } else if !alphabet.contains(&letter) {
                return Err(PFSAError::Syntax("Transitions must have valid letters.".to_string()));
            } else if (next_state <= &0) || (next_state > &states) {
                return Err(PFSAError::Syntax("Transitions must go to valid states.".to_string()));
            } else if (prob < &0.0) || (prob > &1.0) {
                return Err(PFSAError::Syntax("Transitions probability must be a valid probability (between 0 and 1 inclusive).".to_string()));
            }
        }

        //The starting state must be a valid state
        if (start_state <= 0) || (start_state > states) {
            return Err(PFSAError::Syntax("start_state must be a valid state.".to_string()));
        }

        //All accepting state must be valid states
        for accepting_state in &accepting_states {
            if (accepting_state <= &0) || (accepting_state > &states) {
                return Err(PFSAError::Syntax("Accepting states must be valid states.".to_string()));
            }
        }

        for i in 1..(states+1) {
            let mut total_prob: f64  = 0.0;
            for letter in &alphabet {
                //since we have checked every (state,letter) pair we know this keys exists, then we grab the probability associated with it.
                total_prob += transitions.get(&(i,letter.clone())).unwrap().1;
            }

            if accepting_states.contains(&i) {
                //since we know that probabilites are positives we only need to check that they sum to less than one.
                if total_prob >= 1.0 {
                    return Err(PFSAError::Syntax("Outgoing probabilites of an accepting state must sum to less than 1.".to_string()));
                }
            }else {
                //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
                if (1.0 - total_prob).abs() > FLOAT_ERROR_TOLERANCE {
                    return Err(PFSAError::Syntax("Outgoing probabilites of an non-accepting state must sum to 1.".to_string()));
                }
            }
        }

        Ok(ProbabilisticFiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, })
    }

    //getters
    pub fn get_num_of_states(&self) -> usize {
        self.states
    }

    pub fn get_alphabet(&self) -> HashSet<T> {
        self.alphabet.clone()
    }

    pub fn get_start_state(&self) -> usize {
        self.start_state
    }

    pub fn get_accepting_states(&self) -> HashSet<usize> {
        self.accepting_states.clone()
    }

    //step
    //Purpose:
    //    Given a state and a letter, returns a tuple with the resulting state and the probability
    //    of that letter being chosen at the current state.
    //Pre-conditions:
    //    current_state must be a valid state and letter must be in alphabet.
    pub fn step(&self, current_state: usize, letter: T) -> Option<&(usize, f64)> {
        self.transitions.get(&(current_state,letter))
    }

    //get_next_step
    //Purpose:
    //    Returns a Option of a tuple containing a letter and state where the letter is chosen
    //    with the proabilites given by transitions and the state is the state the machine will be in next.
    //    A None value indicates that the pFSA has ended the sequence.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    current_state must be a valid state.
    pub fn get_next_step(&self, current_state: usize) -> Option<(T, usize)> {
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
        //Will only return if key doesn't exist (An error which shouldn't occur since init checks if the key exists) or if the path ended (in which the current
        //  state must be an ending state).
        return None;
    }

    //generate
    //Purpose:
    //    Generates a word obtained and moving through the pFSA.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    None
    pub fn generate(&self) -> Vec<T> {
        let mut current_state = self.start_state;
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
    //    Returns the probability of a given word being the result of the output of the pFSA.
    //    Since floats are used the probability may not be exact.
    //Pre-conditions:
    //    All letters in word must be in the alphabet
    pub fn prob_of_word(&self, word: &Vec<T>) -> Result<f64, PFSAError> {
        let mut current_state = self.start_state; //Need to track current state
        let mut prob_of_word = 1.0;
        for letter in word {
            if !self.alphabet.contains(&letter) {
                return Err(PFSAError::Syntax("Word contains an invalid letter.".to_string()));
            }
            let (next_state,prob_of_letter) = *self.transitions.get(&(current_state,letter.clone())).unwrap();
            current_state = next_state;
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
    fn pfsa_file_errors() {

        //Attempted to read a non-existant file
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("non_existant_file".to_string()).is_err());

        //File with insufficent information to do anything.
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_pFSA_1".to_string()).is_err());

        //States is given as -1 (must be a usize)
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_pFSA_2".to_string()).is_err());

        //Starting_states is given as -1 (must be a usize)
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_pFSA_3".to_string()).is_err());

        //Accepting states has a non-usize entry 'q' (all entries be usize)
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_pFSA_4".to_string()).is_err());

        //A transition line a has a non-usize entry for a state (all states be usize)
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_pFSA_5".to_string()).is_err());

        //A transition line a has a insufficent number of arguments (requires three, a state, a letter, a state, and a probability)
        assert!(ProbabilisticFiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_6".to_string()).is_err());
    }

    #[test]
    fn pfsa_init_errors() {

        //setup
        let a = -1;
        let b = -2;

        let alphabet_1: HashSet<isize> =  [a].iter().cloned().collect();
        let alphabet_2: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states_1: HashSet<usize> = [1].iter().cloned().collect();
        let accepting_states_2: HashSet<usize> = [0].iter().cloned().collect();
        let accepting_states_3: HashSet<usize> = [3].iter().cloned().collect();
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

        //Attempted init with 0 states (states must be a positive integer)
        assert!(ProbabilisticFiniteStateAutomata::init((0,alphabet_2.clone(),trans_1.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with empty alphabet (alphabet must be a non-empty)
        assert!(ProbabilisticFiniteStateAutomata::init((2,HashSet::new(),trans_1.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with start_state at 0 (states must be positive integers)
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),0,accepting_states_1.clone())).is_err());

        //Attempted init with start_state larger than the given number of states (state id may not exceed total number of states.)
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),3,accepting_states_1.clone())).is_err());

        //Attempted init with 0 as an accepting state (states must be positive integers.)
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),2,accepting_states_2.clone())).is_err());

        //Attempted init with an accepting state larger than the given number of states (state id may not exceed total number of states.)
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),2,accepting_states_3.clone())).is_err());

        //Attempted init with a transition matrix with letter not used in alphabet.
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_1.clone(),trans_1.clone(),2,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix not containing all (state,letter) pairs.
        assert!(ProbabilisticFiniteStateAutomata::init((2,alphabet_2.clone(),trans_2.clone(),2,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix containing a transition to a invalid state
        assert!(ProbabilisticFiniteStateAutomata::init((1,alphabet_2.clone(),trans_3.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix containing a probability less than 0.
        assert!(ProbabilisticFiniteStateAutomata::init((1,alphabet_2.clone(),trans_4.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix containing a probability greater than 1.
        assert!(ProbabilisticFiniteStateAutomata::init((1,alphabet_2.clone(),trans_5.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix containing an accepting state for which its outgoing probabilites sum to 1.
        assert!(ProbabilisticFiniteStateAutomata::init((1,alphabet_2.clone(),trans_1.clone(),1,accepting_states_4.clone())).is_err());

        //Attempted init with a transition matrix containing a non-accepting state for which its outgoring porbabilites don't sum to 1.
        assert!(ProbabilisticFiniteStateAutomata::init((1,alphabet_2.clone(),trans_6.clone(),1,accepting_states_1.clone())).is_err());
    }

    #[test]
    fn pfsa_getters() {
        //setup
        let a = -1;
        let b = -2;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), (usize,f64)> =
            [((1,a), (1,0.5)), ((1,b), (2,0.4)), ((2,a), (1,0.5)), ((2,b), (3,0.3)), ((3,a), (1,0.5)), ((3,b), (3,0.5)),].iter().cloned().collect();

        let test_pfsa = ProbabilisticFiniteStateAutomata::init((3,alphabet.clone(),trans,1,accepting_states.clone())).unwrap();

        assert_eq!(test_pfsa.get_num_of_states(),3);

        assert_eq!(test_pfsa.get_alphabet(),alphabet);

        assert_eq!(test_pfsa.get_start_state(),1);

        assert_eq!(test_pfsa.get_accepting_states(),accepting_states);
    }

    #[test]
    fn pfsa_transitions() {
        //setup
        let a = -1;
        let b = -2;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), (usize,f64)> =
            [((1,a), (1,0.5)), ((1,b), (2,0.4)), ((2,a), (1,0.5)), ((2,b), (3,0.3)), ((3,a), (1,0.5)), ((3,b), (3,0.5)),].iter().cloned().collect();

        let test_pfsa = ProbabilisticFiniteStateAutomata::init((3,alphabet.clone(),trans.clone(),1,accepting_states.clone())).unwrap();

        for ((state, letter), (next_state, prob)) in &trans {
            assert_eq!(test_pfsa.step(*state,*letter).unwrap(),&(*next_state, prob.clone()));
        }
    }

    #[test]
    fn pfsa_generate() {

        //setup
        let a = -1;
        let b = -2;

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
        let my_pfsa = ProbabilisticFiniteStateAutomata::init((2,alphabet.clone(),trans_prob,1,final_states.clone())).unwrap();

        for _i in 0..1000 {
            let generated_sequence = my_pfsa.generate();
            assert!(my_fsa.accepts_sequence(&generated_sequence).unwrap());
        }
    }

    #[test]
    fn pfsa_prob_of_word() {

        //setup
        let a = -1;
        let b = -2;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let final_states: HashSet<usize> = [1].iter().cloned().collect();
        let trans_prob: HashMap<(usize, isize), (usize, f64)> =
        [((1,a), (1,0.3)),
        ((1,b), (2,0.3)),
        ((2,a), (1,0.5)),
        ((2,b), (2,0.5)),]
        .iter().cloned().collect();
        let my_pfsa = ProbabilisticFiniteStateAutomata::init((2,alphabet,trans_prob,1,final_states)).unwrap();

        assert!( (my_pfsa.prob_of_word(&vec![]).unwrap() - 0.4).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![a]).unwrap() - 0.12).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![b,a]).unwrap() - 0.06).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![b,b,a]).unwrap() - 0.03).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![a,b,a]).unwrap() - 0.018).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![b,b,b,a]).unwrap() - 0.015).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![b]).unwrap() - 0.0).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (my_pfsa.prob_of_word(&vec![a,a,a,b]).unwrap() - 0.0).abs() < FLOAT_ERROR_TOLERANCE );
    }

}