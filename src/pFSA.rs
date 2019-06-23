//Author: Everett Sullivan
//Date Created: 6/16/2019
//Purpose To create a probabilistic finite state automata (FSA)
//Notes:
//Ideas: The constructors do not test that the transition function (transitions)
//          contains every (state,letter) pair from states X alphabet, or that the
//          proabilites are proper and add up to the correct amount, or that only
//          accepting state may add up to less than one.
//          Should checks be impelemented?
//       The "init_from_file()" function assumes the file is in the correct format.
//          Should checks be impelemented?
//       The functions "step_str" is a &str variant of "step".
//          This was because &str and String are not the same type.
//          However we can use ".to_string()"" to get a &str type into a String type
//          which bypasses the need for the &str variants of the function.
//          Is the &str variant really needed?

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;
use rand::Rng;

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

    //init_from_file
    //Purpose:
    //    Creates a pFSA from a file.
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
    pub fn init_from_file(filename: String) -> ProbabilisticFiniteStateAutomata<String> {
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
            let probability = transition_string[3].parse::<f64>().unwrap();
            transitions.insert((current_state,current_letter),(next_state,probability));
        }

        ProbabilisticFiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, }
    }

    //step_str
    //Purpose:
    //    Given a state and a letter, returns a tuple with the resulting state and the probability
    //    of that letter being chosen at the current state.
    //    Used when the user has a &str and the alphabet is a bunch of Strings.
    //Pre-conditions:
    //    current_state must be a valid state and letter must be in alphabet.
    pub fn step_str(&self, current_state: usize, letter: &str) -> Option<&(usize, f64)> {
        self.transitions.get(&(current_state,letter.to_string()))
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
        states: usize,
        alphabet: HashSet<T>,
        transitions: HashMap<(usize,T),(usize,f64)>,
        start_state: usize,
        accepting_states: HashSet<usize>
    ) -> ProbabilisticFiniteStateAutomata<T> {
        ProbabilisticFiniteStateAutomata{ states, alphabet, transitions, start_state, accepting_states, }
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
        //Will only return if key doesn't exist (An error) or if the path ended (in which the current
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
    pub fn prob_of_word(&self, word: &Vec<T>) -> f64 {
        let mut current_state = self.start_state; //Need to track current state
        let mut prob_of_word = 1.0;
        for letter in word {
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

        prob_of_word*prob_of_ending
    }
}