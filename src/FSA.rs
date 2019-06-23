//Author: Everett Sullivan
//Date Created: 6/15/2019
//Purpose: To create a finite state automata (FSA)
//Notes:
//Ideas: The constructors do not test that the transition function (transitions)
//          contains every (state,letter) pair from states X alphabet.
//          Should checks be impelemented?
//       The "init_from_file()" function assumes the file is in the correct format.
//          Should checks be impelemented?
//       The functions "get_str_result" and "accepts_str_sequence" are &str variants
//          of "get_result" and "accepts_str_sequence".
//          This was because &str and String are not the same type.
//          However we can use ".to_string()"" to get a &str type into a String type,
//          and ".iter().map(|s| s.to_string()).collect::<Vec<String>>()" to turn
//          a Vec<&str> to a Vec<String> which bypasses the need for the &str variants
//          of the functions.
//          Do the &str variants really be needed?

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;

pub struct FiniteStateAutomata<T: Eq + Hash + Clone> {
    states: usize, //Since states are enumerated by positive integers it suffice to
                   //   return the number of states to know what all the states are.
	alphabet: HashSet<T>,
	transitions: HashMap<(usize, T), usize>,
	start_state: usize,
    accepting_states: HashSet<usize>
}

impl FiniteStateAutomata<String> {

    //init_from_file
    //Purpose:
    //    Creates a pFSA from a file.
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
    pub fn init_from_file(filename: String) -> FiniteStateAutomata<String> {
        // Open file.
        let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

        // Break file into lines.
        let lines: Vec<&str> = contents.lines().collect();

        // get number of states
        let states = lines[0].parse::<usize>().unwrap();

        // get alphabet
        let mut alphabet = HashSet::new();
        let alphabet_string = lines[1].split_whitespace().collect::<Vec<&str>>();
        for letter in alphabet_string {
            alphabet.insert(letter.to_string());
        }

        //get starting state
        let start_state = lines[2].parse::<usize>().unwrap();

        //get accepting states
        let mut accepting_states = HashSet::new();
        let accepting_states_string = lines[3].split_whitespace().collect::<Vec<&str>>();
        for state in accepting_states_string {
            accepting_states.insert(state.parse::<usize>().unwrap());
        }

        //get transitions
        let mut transitions = HashMap::new();
        for i in 4..lines.len() {
            let transition_string = lines[i].split_whitespace().collect::<Vec<&str>>();
            let current_state = transition_string[0].parse::<usize>().unwrap();
            let current_letter = transition_string[1].to_string();
            let next_state = transition_string[2].parse::<usize>().unwrap();
            transitions.insert((current_state,current_letter),next_state);
        }

        FiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, }
    }

    //get_str_result
    //Purpose:
    //    Returns the state that will be reached when at state current_state with input letter.
    //    Used when the user has a &str and the alphabet is a bunch of Strings.
    //Pre-conditions:
    //    current_state must be a valid state and letter must be in alphabet.
    pub fn get_str_result(&self, current_state: usize, letter: &str) -> usize {
        if (current_state == 0) || (current_state > self.states) {
            panic!("Given State doesn't exist.")
        } else if !self.alphabet.contains(&letter.to_string()) {
            panic!("Given letter is not the alphabet.")
        }
        //transitions must contain every possible pair of a statte and letter, so it must exist at this point.
        *self.transitions.get(&(current_state, letter.to_string())).unwrap()
    }

    //accepts_str_sequence
    //Purpose:
    //    Returns true if putting the word through the FSA would result in an accepting state.
    //    Used when the user has a vec of &str and the alphabet is a bunch of Strings.
    //Pre-conditions:
    //    All letters in word must be in the alphabet
    pub fn accepts_str_sequence(&mut self, word: &Vec<&str>) -> bool {
        let mut current_state = self.start_state;
        for letter in word {
            current_state = *self.transitions.get(&(current_state,letter.to_string())).unwrap();
        }
        //if the final state is an accepting state return true, else return false.
        self.accepting_states.contains(&current_state)
    }

}

impl <T: Eq + Hash + Clone> FiniteStateAutomata<T> {

    //init_from_file
    //Purpose:
    //    Creates a FSA.
    //Pre-conditions:
    //    states is positive, alphabet is non-empty, transitions has as keys every possible state letter pair.
    pub fn init(
        states: usize,
        alphabet: HashSet<T>,
        transitions: HashMap<(usize, T), usize>,
        start_state: usize,
        accepting_states: HashSet<usize>
    ) -> FiniteStateAutomata<T> {
        FiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, }
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
    pub fn get_result(&self, current_state: usize, letter: T) -> usize {
        if (current_state == 0) || (current_state > self.states) {
            panic!("Given State doesn't exist.")
        } else if !self.alphabet.contains(&letter) {
            panic!("Given letter is not the alphabet.")
        }
        //transitions must contain every possible pair of a statte and letter, so it must exist at this point.
        *self.transitions.get(&(current_state, letter)).unwrap()
    }

    //accepts_sequence
    //Purpose:
    //    Returns true if putting the word through the FSA would result in an accepting state.
    //Pre-conditions:
    //    All letters in word must be in the alphabet
    pub fn accepts_sequence(&self, word: &Vec<T>) -> bool {
        let mut current_state = self.start_state;
        for letter in word {
            current_state = *self.transitions.get(&(current_state,letter.clone())).unwrap();
        }
        //if the final state is an accepting state return true, else return false.
        self.accepting_states.contains(&current_state)
    }
}