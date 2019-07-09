//Author: Everett Sullivan
//Date Created: 6/15/2019
//Purpose: To create a finite state automata (FSA)
//Notes:
//Ideas: 

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;

////////////////////
//Custom Error handling code
////////////////////

#[derive(Debug)]
pub enum FSAError {
    Io(io::Error),
    Parse(num::ParseIntError),
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for FSAError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FSAError::Io(ref err) => err.fmt(f),
            FSAError::Parse(ref err) => err.fmt(f),
            FSAError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for FSAError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            FSAError::Io(ref err) => Some(err),
            FSAError::Parse(ref err) => Some(err),
            FSAError::Syntax(ref _err_string)  => None,
        }
    }
}

use std::io;
use std::num;

// need to turn errors into the custum type, used by the ? operator to convert errors.
impl From<io::Error> for FSAError {
    fn from(err: io::Error) -> FSAError {
        FSAError::Io(err)
    }
}

impl From<num::ParseIntError> for FSAError {
    fn from(err: num::ParseIntError) -> FSAError {
        FSAError::Parse(err)
    }
}

////////////////////
//FSA code
////////////////////

pub struct FiniteStateAutomata<T: Eq + Hash + Clone> {
    states: usize, //Since states are enumerated by positive integers it suffice to
                   //   return the number of states to know what all the states are.
	alphabet: HashSet<T>,
	transitions: HashMap<(usize, T), usize>,
	start_state: usize,
    accepting_states: HashSet<usize>
}

impl FiniteStateAutomata<String> {

    //data_from_file
    //Purpose:
    //    Creates the data needed to construct a FSA from a file.
    //Pre-conditions:
    //    file is in the correct format.
    //Note:
    //  file are arranged as follows:
    //  The first line contains the number of states (must be a positive integer).
    //  The second line contains the alphabet of the FSA where each letter is sperated
    //      by whitespace (must have at least one letter).
    //      (Note that "letter" here just means a element of the alphabet, the "letter"
    //      can be anything abstractly but if we are reading from a file we assume they are strings.)
    //  The third line has the (single) start state
    //  The fourth line contains the accepting states seperated by whitespace
    //  All following lines are the transition instructions.
    //      Each line contains, in order, the state it is starting from, the letter it
    //      just read, and the state it will go to with that input.
    //      Note that every pair of state and letter must have a transition.
    pub fn data_from_file(filename: String) -> Result<(usize, HashSet<String>, HashMap<(usize, String), usize>, usize, HashSet<usize>), FSAError> {
        // Open file.
        let contents = fs::read_to_string(filename)?;

        // Break file into lines.
        let lines: Vec<&str> = contents.lines().collect();

        //A seperate line is needed for number of state, the alphabet, starting state, accepting states, and each transition.
        if lines.len() < 5 {
            return Err(FSAError::Syntax("A FSA file requires at least five lines.".to_string()));
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
            let state_as_usize: usize = state.parse()?;
            accepting_states.insert(state_as_usize);
        }

        //get transitions
        let mut transitions = HashMap::new();
        for i in 4..lines.len() {
            let transition_string = lines[i].split_whitespace().collect::<Vec<&str>>();
            if transition_string.len() != 3 {
                return Err(FSAError::Syntax("A transition requires three parts.".to_string()));
            }

            let current_state: usize = transition_string[0].parse()?;

            let current_letter = transition_string[1].to_string();

            let next_state: usize = transition_string[2].parse()?;

            transitions.insert((current_state,current_letter),next_state);
        }

        Ok((states, alphabet, transitions, start_state, accepting_states))
    }

}

impl <T: Eq + Hash + Clone> FiniteStateAutomata<T> {

    //init
    //Purpose:
    //    Creates a FSA.
    //Pre-conditions:
    //    states is positive, alphabet is non-empty, transitions has as keys every possible state
    //    letter pair, start state is a valid state, and accepting_states contains only valid states.
    pub fn init(
        (states, alphabet, transitions, start_state, accepting_states): (usize, HashSet<T>, HashMap<(usize, T), usize>, usize, HashSet<usize>,)
    ) -> Result<FiniteStateAutomata<T>, FSAError> {

        //There must be at least one state.
        if states <= 0 {
            return Err(FSAError::Syntax("States must be a positive integer.".to_string()));
        }

        //There must be at least one letter in the alphabet
        if alphabet.len() == 0 {
            return Err(FSAError::Syntax("Alphabet must be non-empty.".to_string()));
        }

        //Every (state,letter) pair must be a key in Transitions.
        let numbers = 1..;
        let state_iter = numbers.take(states);
        for (state, letter) in state_iter.zip(alphabet.iter()) {
            if !transitions.contains_key(&(state, letter.clone())) {
                return Err(FSAError::Syntax("Transitions must contain every (state,letter) pair.".to_string()));
            }
        }

        //Every transition in Transitions must be a valid (state,letter) pair.
        for ((state, letter), next_state) in &transitions {
            if (state <= &0) || (state > &states) {
                return Err(FSAError::Syntax("Transitions must be from valid states.".to_string()));
            } else if !alphabet.contains(&letter) {
                return Err(FSAError::Syntax("Transitions must have valid letters.".to_string()));
            } else if (next_state <= &0) || (next_state > &states) {
                return Err(FSAError::Syntax("Transitions must go to valid states.".to_string()));
            }
        }

        //The starting state must be a valid state
        if (start_state <= 0) || (start_state > states) {
            return Err(FSAError::Syntax("start_state must be a valid state.".to_string()));
        }

        //All accepting state must be valid states
        for accepting_state in &accepting_states {
            if (accepting_state <= &0) || (accepting_state > &states) {
                return Err(FSAError::Syntax("Accepting states must be valid states.".to_string()));
            }
        }

        Ok(FiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, })
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

    //get_result
    //Purpose:
    //    Returns the state that will be reached when at state current_state with input letter.
    //Pre-conditions:
    //    current_state must be a valid state and letter must be in alphabet.
    pub fn get_result(&self, current_state: usize, letter: T) -> Result<usize, FSAError> {
        if (current_state == 0) || (current_state > self.states) {
            return Err(FSAError::Syntax("Attempted input of invalid state.".to_string()));
        } else if !self.alphabet.contains(&letter) {
            return Err(FSAError::Syntax("Attempted input of invalid letter.".to_string()));
        }
        //transitions must contain every possible pair of a statte and letter, so it must exist at this point.
        Ok(*self.transitions.get(&(current_state, letter)).unwrap())
    }

    //accepts_sequence
    //Purpose:
    //    Returns true if putting the word through the FSA would result in an accepting state.
    //Pre-conditions:
    //    All letters in word must be in the alphabet
    pub fn accepts_sequence(&self, word: &Vec<T>) -> Result<bool, FSAError> {
        let mut current_state = self.start_state;
        for letter in word {
            if !self.alphabet.contains(&letter) {
                return Err(FSAError::Syntax("Word contains an invalid letter.".to_string()));
            }
            //transitions must contain every possible pair of a statte and letter, so it must exist at this point.
            current_state = *self.transitions.get(&(current_state,letter.clone())).unwrap();
        }
        //if the final state is an accepting state return true, else return false.
        Ok(self.accepting_states.contains(&current_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsa_file_errors() {

        //Attempted to read a non-existant file
        assert!(FiniteStateAutomata::data_from_file("non_existant_file".to_string()).is_err());

        //File with insufficent information to do anything.
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_1".to_string()).is_err());

        //States is given as -1 (must be a usize)
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_2".to_string()).is_err());

        //Starting_states is given as -1 (must be a usize)
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_3".to_string()).is_err());

        //Accepting states has a non-usize entry 'q' (all entries be usize)
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_4".to_string()).is_err());

        //A transition line a has a non-usize entry for a state (all states be usize)
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_5".to_string()).is_err());

        //A transition line a has a insufficent number of arguments (requires three, a state, a letter, and a state)
        assert!(FiniteStateAutomata::data_from_file("..\\test_files\\test_FSA_6".to_string()).is_err());
    }

    #[test]
    fn fsa_init_errors() {

        //setup
        let a = -1;
        let b = -2;

        let alphabet_1: HashSet<isize> =  [a].iter().cloned().collect();
        let alphabet_2: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states_1: HashSet<usize> = [1].iter().cloned().collect();
        let accepting_states_2: HashSet<usize> = [0].iter().cloned().collect();
        let accepting_states_3: HashSet<usize> = [3].iter().cloned().collect();
        let trans_1: HashMap<(usize, isize), usize> =
            [((1,a), 1),((1,b), 2),((2,a), 1),((2,b), 2),].iter().cloned().collect();
        let trans_2: HashMap<(usize, isize), usize> =
            [((1,a), 1),((1,b), 2),].iter().cloned().collect();
        let trans_3: HashMap<(usize, isize), usize> =
            [((1,a), 1),((1,b), 2),].iter().cloned().collect();

        //Attempted init with 0 states (states must be a positive integer)
        assert!(FiniteStateAutomata::init((0,alphabet_2.clone(),trans_1.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with empty alphabet (alphabet must be a non-empty)
        assert!(FiniteStateAutomata::init((2,HashSet::new(),trans_1.clone(),1,accepting_states_1.clone())).is_err());

        //Attempted init with start_state at 0 (states must be positive integers)
        assert!(FiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),0,accepting_states_1.clone())).is_err());

        //Attempted init with start_state larger than the given number of states (state id may not exceed total number of states.)
        assert!(FiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),3,accepting_states_1.clone())).is_err());

        //Attempted init with 0 as an accepting state (states must be positive integers.)
        assert!(FiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),2,accepting_states_2.clone())).is_err());

        //Attempted init with an accepting state larger than the given number of states (state id may not exceed total number of states.)
        assert!(FiniteStateAutomata::init((2,alphabet_2.clone(),trans_1.clone(),2,accepting_states_3.clone())).is_err());

        //Attempted init with a transition matrix with letter not used in alphabet.
        assert!(FiniteStateAutomata::init((2,alphabet_1.clone(),trans_1.clone(),2,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix not containing all (state,letter) pairs.
        assert!(FiniteStateAutomata::init((2,alphabet_2.clone(),trans_2.clone(),2,accepting_states_1.clone())).is_err());

        //Attempted init with a transition matrix containg a transition to a invalid state
        assert!(FiniteStateAutomata::init((1,alphabet_2.clone(),trans_3.clone(),2,accepting_states_1.clone())).is_err());  
    }

    #[test]
    fn fsa_getters() {
        //setup
        let a = -1;
        let b = -2;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), usize> =
            [((1,a), 1), ((1,b), 2), ((2,a), 1), ((2,b), 3), ((3,a), 1), ((3,b), 3),].iter().cloned().collect();

        let test_fsa = FiniteStateAutomata::init((3,alphabet.clone(),trans,1,accepting_states.clone())).unwrap();

        assert_eq!(test_fsa.get_num_of_states(),3);

        assert_eq!(test_fsa.get_alphabet(),alphabet);

        assert_eq!(test_fsa.get_start_state(),1);

        assert_eq!(test_fsa.get_accepting_states(),accepting_states);
    }

    #[test]
    fn fsa_transitions() {
        //setup
        let a = -1;
        let b = -2;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), usize> =
            [((1,a), 1), ((1,b), 2), ((2,a), 1), ((2,b), 3), ((3,a), 1), ((3,b), 3),].iter().cloned().collect();

        let test_fsa = FiniteStateAutomata::init((3,alphabet.clone(),trans.clone(),1,accepting_states.clone())).unwrap();

        for ((state, letter), next_state) in &trans {
            assert_eq!(test_fsa.get_result(*state,*letter).unwrap(),*next_state);
        }


        //get_result should return an error if state is given as 0.
        assert!(test_fsa.get_result(0,-1).is_err());

        //get_result should return an error if state is larger than the number of states.
        assert!(test_fsa.get_result(4,-1).is_err());

        //get_result should return an error if letter is not in the alphabet.
        assert!(test_fsa.get_result(1,-3).is_err());
    }

    #[test]
    fn fsa_accepts() {
        //setup
        let a = -1;
        let b = -2;
        let c = -3;

        let alphabet: HashSet<isize> =  [a,b].iter().cloned().collect();
        let accepting_states: HashSet<usize> = [1,2].iter().cloned().collect();
        let trans: HashMap<(usize, isize), usize> =
            [((1,a), 1), ((1,b), 2), ((2,a), 1), ((2,b), 3), ((3,a), 1), ((3,b), 3),].iter().cloned().collect();

        //This FSA should only accepting strings that don't end with two b's in a row.
        let test_fsa = FiniteStateAutomata::init((3,alphabet.clone(),trans.clone(),1,accepting_states.clone())).unwrap();

        //these sequences should be accepted
        assert!(test_fsa.accepts_sequence(&vec![]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![b]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![b,a]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![b,b,a]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![a,a,a,a,a]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![b,b,b,b,b,b,a]).unwrap());
        assert!(test_fsa.accepts_sequence(&vec![b,b,a,b]).unwrap());

        //these sequences should be rejected
        assert!(!test_fsa.accepts_sequence(&vec![b,b]).unwrap());
        assert!(!test_fsa.accepts_sequence(&vec![a,b,b]).unwrap());
        assert!(!test_fsa.accepts_sequence(&vec![b,b,b]).unwrap());

        //this sequence should cause an error
        assert!(test_fsa.accepts_sequence(&vec![b,c]).is_err());
    }

}
