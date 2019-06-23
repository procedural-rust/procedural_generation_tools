//Author: Everett Sullivan
//Date Created: 6/16/2019
//Purpose To create an Abstract probabilistic finite state automata (FSA)
//Notes: Another Implementation of pFSA except that the state are a set of
//          objects (which may be different then the alphabet).
//          This was specifally created for ngrams.rs but can be used for other applications.
//Ideas: The constructors do not test that the transition function (transitions)
//          contains every (state,letter) pair from states X alphabet, or that the
//          proabilites are proper and add up to the correct amount, or that only
//          accepting state may add up to less than one.
//          Should checks be impelemented?

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use rand::Rng;

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
    pub fn prob_of_word(&self, word: &Vec<T>) -> f64 {
        let mut current_state = self.start_state.clone(); //Need to track current state
        let mut prob_of_word = 1.0;
        for letter in word {
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

        prob_of_word*prob_of_ending
    }
}