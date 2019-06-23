//Author: Everett Sullivan
//Date Created: 6/15/2019
//Purpose To create a context free grammer (CFG)
//Notes:
//Ideas: The constructors do not test that every variable has a rule or that the
//          proabilites are proper and add up to the correct amount.
//          Should checks be impelemented?
//       The "init_from_file()" function assumes the file is in the correct format.
//          Should checks be impelemented?

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;
use rand::Rng;

pub struct ContextFreeGrammar<T: Eq + Hash + Clone> {
    variables: HashSet<T>,
    terminals: HashSet<T>,
	rules: HashMap<T,Vec<(Vec<T>, f32)>>,
	start_variable: T,
}

impl ContextFreeGrammar<String> {

    //init_from_file
    //Purpose:
    //    Creates a CFG from a file.
    //Pre-conditions:
    //    file is in the correct format.
    //Note:
    //  file are arranged as follows:
    //  The first line contains the variables of the CFG where each variable is sperated by whitespace (must have at least one variable).
    //      (Note that "variable" here just means a element that appears in the left hand side of a rule, the "variable"
    //      can be anything abstractly but if wer are reading from a file we assume they are strings.)
    //  The second line contains the terminals of the CFG where each terminals is sperated by whitespace
    //      (a CFG can have no termials, but then it is a very boring CFG).
    //  The thrid line has the (single) start variable.
    //  All following lines are the rules.
    //      Each line contains, in order, the variable, the probability of yielding the following rule, and the sequence of
    //      variables and terminals that replace the given variable from the rule.
    //      (Every variable must have at least one rule where that transform it, the yield may be empty.)
    pub fn init_from_file(filename: String) -> ContextFreeGrammar<String> {
        // Open file.
        let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

        // Break file into lines.
        let lines: Vec<&str> = contents.lines().collect();

        // get variables
        let mut variables = HashSet::new();
        let variables_string = lines[0].split_whitespace().collect::<Vec<&str>>();
        for variable in variables_string {
            variables.insert(variable.to_string());
        }

        //get terminals
        let mut terminals = HashSet::new();
        let terminals_string = lines[1].split_whitespace().collect::<Vec<&str>>();
        for terminal in terminals_string {
            terminals.insert(terminal.to_string());
        }

        //get starting variable
        let start_variable = lines[2].to_string();

        //get rules
        let mut rules: HashMap<String,Vec<(Vec<String>, f32)>> = HashMap::new();
        for i in 3..lines.len() {
            let rule_string = lines[i].split_whitespace().collect::<Vec<&str>>();
            let current_variable = rule_string[0].to_string();
            let probability = rule_string[1].parse::<f32>().unwrap();
            //The remaining part of the line is the yield
            let the_yield: Vec<String> = rule_string[2..rule_string.len()].iter().cloned().map(|s| s.to_string()).collect();
            match rules.remove_entry(&current_variable) {
                Some((key,mut result)) => {
                    //Add the yield to the possible yields
                    result.push((the_yield,probability));
                    //reinsert key
                    rules.insert(key,result);
                },
                None => {
                    //If the key doesn't exist then this is the first rule with that variable
                    rules.insert(current_variable,vec![(the_yield,probability)]);
                },
            }
        }

        ContextFreeGrammar{ variables, terminals, rules, start_variable, }
    }

}

impl <T: Eq + Hash + Clone> ContextFreeGrammar<T> {

    //init
    //Purpose:
    //    Creates a CFG.
    //Pre-conditions:
    //    variables is non-empty, rules as a key for every variable.
    pub fn init(
        variables: HashSet<T>,
        terminals: HashSet<T>,
        rules: HashMap<T,Vec<(Vec<T>, f32)>>,
        start_variable: T,
    ) -> ContextFreeGrammar<T> {
        ContextFreeGrammar{ variables, terminals, rules, start_variable, }
    }

    //getters
    pub fn get_variables(&self) -> HashSet<T> {
        self.variables.clone()
    }

    pub fn get_terminals(&self) -> HashSet<T> {
        self.terminals.clone()
    }

    pub fn get_start_variable(&self) -> T {
        self.start_variable.clone()
    }

    //get_yield
    //Purpose:
    //    Returns a tuple containing of a yield from a rule with the given variable
    //    with the proabilites given by rules.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    current_variable must be a valid variable.
    pub fn get_yield(&self, current_variable: &T) -> Vec<T> {
        let mut threshold = rand::thread_rng().gen_range(0.0, 1.0);
        let possible_yields = self.rules.get(&current_variable).unwrap();
        for (my_yield,probability) in possible_yields {
            if threshold < *probability {
                return my_yield.clone();
            } else {
                threshold = threshold - probability;
            }
        }

        return Vec::new();
    }

    //get_first_variable
    //Purpose:
    //    Given a word of variables and terminals return a Option of the first index
    //    where a variable appears.
    //    If there are no variables it returns none.
    //    Used for the function generate
    //Pre-conditions:
    //    current_word must contain valid variables and terminals.
    fn get_first_variable(&self, current_word: &Vec<T>) -> Option<usize> {
        for i in 0..current_word.len() {
            if self.variables.contains(&current_word[i]) {
                return Some(i);
            }
        }
        return None;
    }

    //insert_yield
    //Purpose:
    //    Given a word of variables and terminals and an index at which that word has
    //    a variable returns a new word where the given index had a rule applied to it.
    //Pre-conditions:
    //    current_word must contain valid variables and terminals, current_word[index] must be a variable.
    fn insert_yield(&self, current_word: &Vec<T>, index: usize) -> Vec<T> {
        let variable_yield = self.get_yield(&current_word[index]);
        let before_variable = &current_word[0..index];
        let after_variable = &current_word[(index + 1)..current_word.len()];
        let new_word = before_variable.iter().cloned().chain(variable_yield.iter().cloned().chain(after_variable.iter().cloned())).collect();
        return new_word;
    }

    //generate
    //Purpose:
    //    Creates a word of terminals by repeatedly applying rules to variables starting
    //    with the starting variable until there are no variables left.
    //    Note that the output of this function will change after each call.
    //Pre-conditions:
    //    None
    pub fn generate(&self) -> Vec<T> {
        let mut current_word = vec![self.start_variable.clone()];
        loop {
            match self.get_first_variable(&current_word) {
                Some(index) => {
                    current_word = self.insert_yield(&current_word,index);
                },
                None => {
                    return current_word;
                },
            }
        }
    }
}