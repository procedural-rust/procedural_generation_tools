//Author: Everett Sullivan
//Date Created: 6/15/2019
//Purpose To create a context free grammer (CFG)
//Notes:
//Ideas:

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
pub enum CFGError {
    Io(io::Error),
    ParseFloat(num::ParseFloatError),
    Syntax(String),
}

use std::fmt;
use std::error::Error;

impl fmt::Display for CFGError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CFGError::Io(ref err) => err.fmt(f),
            CFGError::ParseFloat(ref err) => err.fmt(f),
            CFGError::Syntax(ref err_string) => write!(f,"{}",err_string),
        }
    }
}

impl Error for CFGError {
    fn cause(&self) -> Option<&Error> {
        match *self {
            CFGError::Io(ref err) => Some(err),
            CFGError::ParseFloat(ref err) => Some(err),
            CFGError::Syntax(ref _err_string)  => None,
        }
    }
}

use std::io;
use std::num;

// need to turn errors into the custum type, used by the ? operator to convert errors.
impl From<io::Error> for CFGError {
    fn from(err: io::Error) -> CFGError {
        CFGError::Io(err)
    }
}

impl From<num::ParseFloatError> for CFGError {
    fn from(err: num::ParseFloatError) -> CFGError {
        CFGError::ParseFloat(err)
    }
}

////////////////////
//CFG code
////////////////////

pub struct ContextFreeGrammar<T: Eq + Hash + Clone> {
    variables: HashSet<T>,
    terminals: HashSet<T>,
	rules: HashMap<T,Vec<(Vec<T>, f64)>>,
	start_variable: T,
}

impl ContextFreeGrammar<String> {

    //data_from_file
    //Purpose:
    //    Creates the data needed to construct a CFG from a file.
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
    pub fn data_from_file(filename: String) -> Result<(HashSet<String>, HashSet<String>, HashMap<String,Vec<(Vec<String>, f64)>>, String), CFGError> {
        // Open file.
        let contents = fs::read_to_string(filename)?;

        // Break file into lines.
        let lines: Vec<&str> = contents.lines().collect();

        //A seperate line is needed for variables, terminals, the starting variable, and 
        if lines.len() < 4 {
            return Err(CFGError::Syntax("A CFG file requires at least four lines.".to_string()));
        }

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
        let mut rules: HashMap<String,Vec<(Vec<String>, f64)>> = HashMap::new();
        for i in 3..lines.len() {
            let rule_string = lines[i].split_whitespace().collect::<Vec<&str>>();
            if rule_string.len() < 2 {
                return Err(CFGError::Syntax("A rule requires at least two parts.".to_string()));
            }
            let current_variable = rule_string[0].to_string();
            let probability = rule_string[1].parse::<f64>()?;
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

        Ok((variables, terminals, rules, start_variable,))
    }

}

impl <T: Eq + Hash + Clone> ContextFreeGrammar<T> {

    //init
    //Purpose:
    //    Creates a CFG.
    //Pre-conditions:
    //    variables is non-empty, rules as a key for every variable.
    pub fn init(
        (variables, terminals, rules, start_variable): (HashSet<T>, HashSet<T>, HashMap<T,Vec<(Vec<T>, f64)>>, T)
    ) -> Result<ContextFreeGrammar<T>, CFGError> {

        //There must be at least one variable
        if variables.len() == 0 {
            return Err(CFGError::Syntax("The set of variables must be non-empty.".to_string()));
        }

        //There must be at least one terminal
        if terminals.len() == 0 {
            return Err(CFGError::Syntax("The set of terminals must be non-empty.".to_string()));
        }

        let intersection: HashSet<_> = variables.intersection(&terminals).collect();
        if intersection.len() != 0 {
            return Err(CFGError::Syntax("Symbols may belong to variables or terminals, but not both.".to_string()));
        }

        //start_variable must be a variable
        if !variables.contains(&start_variable) {
            return Err(CFGError::Syntax("The start variable must belong to the set of variables.".to_string()));
        }

        //check that every variable in variables has at least one rule.
        for variable in &variables {
            if !rules.contains_key(&variable) {
                return Err(CFGError::Syntax("Rules must contain a rule for every variable.".to_string()));
            }
        }

        //Every transition in Transitions must be a valid (state,letter) pair.
        for (variable, yields) in &rules {
            let mut total_prob = 0.0;
            //check that the variable given is in the set of variables.
            if !variables.contains(&variable) {
                return Err(CFGError::Syntax("Rules must be derived from the set variables.".to_string()));
            } else {
                for (sequence, prob) in yields {
                    //check that probabilities are valid
                    if (prob < &0.0) || (prob > &1.0) {
                        return Err(CFGError::Syntax("Probabilities must be valid (between 0 and 1 inclusive).".to_string()));
                    }
                    total_prob += prob;

                    //check that the yields contain only variables and terminals.
                    for symbol in sequence {
                        if !variables.contains(&symbol) && !terminals.contains(&symbol) {
                            return Err(CFGError::Syntax("Yields must contain only variables and terminals.".to_string()));
                        }
                    }
                }

                //testing of equality of floats is tricky, so instead we just check that they are 'close enough'.
                if (1.0 - total_prob).abs() > FLOAT_ERROR_TOLERANCE {
                    return Err(CFGError::Syntax("Outgoing probabilites of rules for a given variable must add to 1.".to_string()));
                }
            }
        }

        Ok(ContextFreeGrammar{ variables, terminals, rules, start_variable, })
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
    fn get_yield(&self, current_variable: &T) -> Vec<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_file_errors() {

        //Attempted to read a non-existant file
        assert!(ContextFreeGrammar::data_from_file("non_existant_file".to_string()).is_err());

        //File with insufficent information to do anything.
        assert!(ContextFreeGrammar::data_from_file("..\\test_files\\test_CFG_1".to_string()).is_err());

        //A transition line a has a insufficent number of arguments (requires three, a state, a letter, a state, and a probability)
        assert!(ContextFreeGrammar::data_from_file("..\\test_files\\test_CFG_2".to_string()).is_err());

        //A transition line has a non-float vaule for the probability of a rule.
        assert!(ContextFreeGrammar::data_from_file("..\\test_files\\test_CFG_3".to_string()).is_err());
    }

    #[test]
    fn cfg_init_errors() {

        //setup
        let s = -1;
        let x = -2;
        let l = -3;
        let r = -4;

        let variables: HashSet<isize> = [s].iter().cloned().collect();
        let terminals_1: HashSet<isize> = [l,r].iter().cloned().collect();
        let terminals_2: HashSet<isize> = [s,l,r].iter().cloned().collect();
        let rules_1: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],0.2), (vec![l,s,r],0.5), (vec![],0.3)])].iter().cloned().collect();
        let rules_2: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],0.2), (vec![l,s,r],0.5), (vec![],0.3)]), (x, vec![(vec![l,r],1.0)])].iter().cloned().collect();
        let rules_3: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],-0.2), (vec![l,s,r],0.5), (vec![],0.3)])].iter().cloned().collect();
        let rules_4: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],0.3), (vec![l,s,r],0.5), (vec![],0.3)])].iter().cloned().collect();
        let rules_5: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],0.3), (vec![l,x,r],0.5), (vec![],0.3)])].iter().cloned().collect();

        //Attempted init with no variables (must have at least one variable)
        assert!(ContextFreeGrammar::init((HashSet::new(),terminals_1.clone(),rules_1.clone(),s)).is_err());

        //Attempted init with no terminals (must have at least one terminal)
        assert!(ContextFreeGrammar::init((variables.clone(),HashSet::new(),rules_1.clone(),s)).is_err());

        //Attempted init with overlaping variables and terminals (no symbol may be both a variable and terminal)
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_2.clone(),rules_1.clone(),s)).is_err());

        //Attempted init with a terminal as the start variable (start variable must be a variable)
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),rules_1.clone(),l)).is_err());

        //Attempted init with rules not containing all variables
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),HashMap::new(),s)).is_err());

        //Attempted init with rules containing extra variables
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),rules_2.clone(),s)).is_err());

        //Attempted init with a rule with invalid probability
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),rules_3.clone(),s)).is_err());

        //Attempted init with a variable whose rules' probabilities add to more than 1.
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),rules_4.clone(),s)).is_err());

        //Attempted init with a yield that contains a non-terminal non-variable symbol.
        assert!(ContextFreeGrammar::init((variables.clone(),terminals_1.clone(),rules_5.clone(),s)).is_err());
    }

    #[test]
    fn cfg_getters() {

        //setup
        let s = -1;
        let l = -2;
        let r = -3;

        let variables: HashSet<isize> = [s].iter().cloned().collect();
        let terminals: HashSet<isize> = [l,r].iter().cloned().collect();
        let rules: HashMap<isize, Vec<(Vec<isize>, f64)>> =
            [(s, vec![(vec![s,s],0.2), (vec![l,s,r],0.5), (vec![],0.3)])].iter().cloned().collect();

        let test_cfg = ContextFreeGrammar::init((variables.clone(),terminals.clone(),rules,s)).unwrap();

        assert_eq!(test_cfg.get_variables(),variables);

        assert_eq!(test_cfg.get_terminals(),terminals);

        assert_eq!(test_cfg.get_start_variable(),s);
    }

}