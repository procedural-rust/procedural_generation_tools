//Author: Everett Sullivan
//Date Created: 6/18/2019
//Purpose To create n-gram Models
//Notes:

use crate::abstractpfsa::AbstractProbabilisticFiniteStateAutomata;

use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::fs;
use std::str;
use regex::Regex;

const FLOAT_ERROR_TOLERANCE: f64 = 0.001;

////////////////////
//n-grams code
////////////////////

//Token
//Purpose:
//  Allows the n-gram algoirthm to indentify start and end markers
//      without having to reserve a T type to signify such.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
enum Token<T: Eq + Hash + Clone> {
    Start,
    Something(T),
    End,
}

#[derive(Debug)]
pub struct NGramModel<T: Eq + Hash + Clone> {
    corpus: Vec<Vec<T>>,
    n: usize,
	model: AbstractProbabilisticFiniteStateAutomata<Vec<Token<T>>,T>,
    ngram_counts: HashMap<Vec<Token<T>>,usize>,
    n1gram_counts: HashMap<Vec<Token<T>>,usize>,
}

impl <T: Eq + Hash + Clone> NGramModel<T> {

    //init
    //Purpose:
    //    Creates a n-gram model from a given corpus.
    //Pre-conditions:
    //    n is positive.
    pub fn init(
        corpus: &Vec<Vec<T>>,
        n: usize,
    ) -> NGramModel<T> {
        let mut ngrams: HashMap<Vec<Token<T>>,usize> = HashMap::new();
        let mut n1grams: HashMap<Vec<Token<T>>,usize> = HashMap::new();
        let mut adjusted_corpus: Vec<Vec<Token<T>>> = Vec::new();

        //prepare corpus for algorithm
        for sequence in corpus.clone() {
            //need to attach (n-1) start tokens to the beging of each line.
            let mut adjusted_sequence = Vec::new();
            for _i in 0..(n-1) {
                adjusted_sequence.push(Token::Start);
            }
            for item in sequence {
                adjusted_sequence.push(Token::Something(item));
            }
            //Only need to attach 1 end token, since as soon as it is encounter we know we have an end state.
            adjusted_sequence.push(Token::End);
            adjusted_corpus.push(adjusted_sequence);
        }

        let mut states: HashSet<Vec<Token<T>>> = HashSet::new();
        let mut alphabet: HashSet<T> = HashSet::new();
        let mut start_state: Vec<Token<T>> = Vec::new();
        let mut accepting_states: HashSet<Vec<Token<T>>> = HashSet::new();

        //Set start state to be n-1 copies of the start token.
        for _i in 0..(n-1) {
            start_state.push(Token::Start);
        }

        //get counts for n-grams and (n-1)-grams.
        //Needed to compute the probability of what comes next.
        for sequence in adjusted_corpus {
            for i in 0..(sequence.len() - n + 1) {
                let ngram = &sequence[i..(i+n)];
                match ngrams.remove_entry(ngram) {
                    Some((_,count)) => {
                        //increase count
                        ngrams.insert(ngram.iter().cloned().collect(),count+1);
                    },
                    None => {
                        //If the key doesn't exist then this is the first rule with that variable
                        ngrams.insert(ngram.iter().cloned().collect(),1);
                    },
                }

                let n1gram = &sequence[i..(i+n-1)];
                match n1grams.remove_entry(n1gram) {
                    Some((_,count)) => {
                        //increase count
                        n1grams.insert(n1gram.iter().cloned().collect(),count+1);
                    },
                    None => {
                        //If the key doesn't exist then this is the first rule with that variable
                        n1grams.insert(n1gram.iter().cloned().collect(),1);
                    },
                }
            }
        }

        let mut transitions: HashMap<(Vec<Token<T>>,T),(Vec<Token<T>>,f64)> = HashMap::new();
        for (ngram, count) in &ngrams {
            let mut before = Vec::new();
            let mut after = Vec::new();
            for i in 0..(n-1) {
                before.push(ngram[i].clone());
                after.push(ngram[i+1].clone());
            }
            let label = &ngram[n-1];
            states.insert(before.clone()); // The (n-1)-grams are the states.

            match label {
                Token::End => { // if the last token is an end token the the first (n-1) tokens form an end state.
                    accepting_states.insert(before);
                }
                Token::Something(true_label) => {
                    alphabet.insert(true_label.clone());
                    //compute probability of transition
                    let prob = (*count as f64) * 1.0 / (*n1grams.get(&before).unwrap() as f64);
                    transitions.insert((before, true_label.clone()),(after,prob));
                }
                _ => {}
            }
        }

        let model = AbstractProbabilisticFiniteStateAutomata::unsafe_init(states,alphabet,transitions,start_state,accepting_states);
        NGramModel{ corpus: corpus.clone(), n: n, model: model, ngram_counts: ngrams, n1gram_counts: n1grams }
    }

    //generate
    //Purpose:
    //    Creates a n-gram model from a given corpus.
    //Pre-conditions:
    //    n is positive.
    pub fn generate(&self) -> Vec<T> {
        self.model.generate()
    }

    //probability
    //Purpose:
    //    Returns the probability of a sequence in this language model.
    //    setting smoothing higher than 0.0 applies laplace smoothing.
    //    new_words should be set to the number of expected novel words (not
    //    observed in the training data) in the test set or other application
    //Pre-conditions:
    //    smoothing is between 0.0 and 1.0
    pub fn probability(&self, sequence: Vec<T>, smoothing: f64, novel_words: usize) -> f64 {

        //tokenize sequence for comparision.
        let mut token_sequence: Vec<Token<T>> = Vec::new();
        for _i in 0..(self.n-1) {
            token_sequence.push(Token::Start);
        }
        for item in sequence {
            token_sequence.push(Token::Something(item));
        }
        token_sequence.push(Token::End);

        //get total number of possible words
        let size_of_word_set = self.model.get_alphabet().len() + novel_words;

        //find probability of the sequence.
        let mut prob_of_sequence = 1.0;
        for i in 0..(token_sequence.len()-self.n + 1) {
            let ngram = &token_sequence[i..(i+self.n)];
            let n1gram = &token_sequence[i..(i+self.n-1)];
            let ngram_count;
            match self.ngram_counts.get(ngram) {
                Some(count) => {
                    ngram_count = *count;
                },
                None => {
                    ngram_count = 0;
                },
            }

            if (ngram_count as f64) + smoothing == 0.0 {
                return 0.0;
            }else{
                prob_of_sequence = prob_of_sequence * ((self.ngram_counts[ngram] as f64) + smoothing * 1.0)/ ((self.n1gram_counts[n1gram] as f64) + smoothing * (size_of_word_set as f64));
            }
        }

        return prob_of_sequence;
    }
}

////////////////
//The following functions are used to turn a file into corpera of sentences
////////////////

//open_as_lines
//Purpose:
//    Given a filename returns a vector of lines determined by the positiion of line breaks.
//    The lines themeselves are vectors of Strings.
//    Removes capitalization and punctuation.
//Pre-conditions:
//    The given file exists.
pub fn open_as_lines(filename: String) -> Vec<Vec<String>> {
    // Open file.
    let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

    // Break file into lines.
    let lines: Vec<&str> = contents.lines().collect();

    let mut corpus: Vec<Vec<String>> = Vec::new();
    let re = Regex::new(r"[.!?;:,]").unwrap();
    for line in lines {
        let lower_line = line.to_ascii_lowercase();
        let clean_line = re.replace_all(&lower_line,"");
        corpus.push(clean_line.split_whitespace().collect::<Vec<&str>>().iter().cloned().map(|s| s.to_string()).collect());
    }

    return corpus;
}

//open_as_sentences
//Purpose:
//    Given a filename returns a vector of sentences determined by the positiion of [.?!].
//    The sentences themeselves are vectors of Strings.
//    Removes capitalization and punctuation.
//Pre-conditions:
//    The given file exists.
pub fn open_as_sentences(filename: String) -> Vec<Vec<String>> {
    // Open file.
    let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

    //Regex to break over sentences
    let re_sentence = Regex::new(r"[.!?]").unwrap();

    let mut corpus: Vec<Vec<String>> = Vec::new();
    let re_punc = Regex::new(r"[.!?;:,]").unwrap();

    // Break file into lines.
    for sentence in re_sentence.split(&contents) {
        let lower_sentence = sentence.to_ascii_lowercase();
        let clean_sentence = re_punc.replace_all(&lower_sentence,"");
        corpus.push(clean_sentence.split_whitespace().collect::<Vec<&str>>().iter().cloned().map(|s| s.to_string()).collect());
    }

    return corpus;
}

//open_as_words
//Purpose:
//    Given a filename returns a vector of words determined by the positiion of whitespace.
//    The words themeselves are vectors of Strings.
//    Removes capitalization and punctuation.
//Pre-conditions:
//    The given file exists.
pub fn open_as_words(filename: String) -> Vec<Vec<char>> {
    // Open file.
    let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");

    let mut corpus: Vec<Vec<char>> = Vec::new();
    let re_punc = Regex::new(r"[.!?;':,]").unwrap();

    let lower_contents = contents.to_ascii_lowercase();
    let clean_contents = re_punc.replace_all(&lower_contents,"");

    // Break file into lines.
    for word in clean_contents.split_whitespace() {
        corpus.push(word.chars().collect::<Vec<char>>().iter().cloned().collect());
    }

    return corpus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_gram_generate() {

        let my_corpus = vec![vec![1,2,3,4,5],vec![5,4,3,2,1]];
        let number_1gram = NGramModel::init(&my_corpus,1);
        let number_2gram = NGramModel::init(&my_corpus,2);
        let number_3gram = NGramModel::init(&my_corpus,3);

        assert!( (number_1gram.probability(vec![],0.0,0) - (1.0/6.0)).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_1gram.probability(vec![1],0.0,0) - (1.0/36.0)).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_1gram.probability(vec![7],0.0,0) - 0.0).abs() < FLOAT_ERROR_TOLERANCE );

        assert!( (number_2gram.probability(vec![1],0.0,0) - 0.25).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_2gram.probability(vec![1,2,1],0.0,0) - 0.0625).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_2gram.probability(vec![1,2,3,4,5],0.0,0) - 0.015625).abs() < FLOAT_ERROR_TOLERANCE );

        assert!( (number_3gram.probability(vec![1,2,3,4,5],0.0,0) - 0.5).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_3gram.probability(vec![5,4,3,2,1],0.0,0) - 0.5).abs() < FLOAT_ERROR_TOLERANCE );
        assert!( (number_3gram.probability(vec![1,2,3,2,1],0.0,0) - 0.0).abs() < FLOAT_ERROR_TOLERANCE );

    }

}