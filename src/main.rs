//Author: Everett Sullivan
//Date Created: 6/15/2019
//Purpose: To expeirment with FSAs pFSAs and CFGs.

use std::collections::{HashSet, HashMap};

mod fsa;
use fsa::FiniteStateAutomata;

mod pfsa;
use pfsa::ProbabilisticFiniteStateAutomata;

mod abstractpfsa;
use abstractpfsa::AbstractProbabilisticFiniteStateAutomata;

mod cfg;
use cfg::ContextFreeGrammar;

mod ngrams;
use ngrams::NGramModel;

mod hmm;
use hmm::HiddenMarkovModel;

fn main() {

    fsa_example();

    cfg_example();

    ngram_example();

    hmm_example();
}

fn vec_to_string(vec_of_str: &Vec<&str>) -> Vec<String> {
    vec_of_str.clone().iter().map(|s| s.to_string()).collect::<Vec<String>>()
}

fn fsa_example() {

    //this fsa accepts only strings that end in b, i.e. [ab]*b.
    let my_fsa1 = FiniteStateAutomata::init_from_file("exampleFSA1.txt".to_string());

    //this fsa accepts only strings that don't end in b (including the empty string), i.e. [ab]*a|e.
    let my_fsa2 = FiniteStateAutomata::init_from_file("exampleFSA2.txt".to_string());

    //this pfsa produces only strings that end in b, i.e. [ab]*b.
    let my_pfsa1 = ProbabilisticFiniteStateAutomata::init_from_file("examplepFSA1.txt".to_string());

    //this pfsa  only strings that don't end in b (including the empty string), i.e. [ab]*a|e.
    let my_pfsa2 = ProbabilisticFiniteStateAutomata::init_from_file("examplepFSA2.txt".to_string());

    let mut alphabet = HashSet::new();
    alphabet.insert("a".to_string());
    alphabet.insert("b".to_string());
    let mut final_states = HashSet::new();
    final_states.insert(2);
    let trans: HashMap<(usize, String), usize> =
    [((1,"a".to_string()), 1),
     ((1,"b".to_string()), 2),
     ((2,"a".to_string()), 1),
     ((2,"b".to_string()), 2),]
     .iter().cloned().collect();
    let mut my_fsa1_manual = FiniteStateAutomata::init(2,alphabet,trans,1,final_states);

    println!("\nOutput from FSA 1\n");

    for _i in 0..20 {
        let generated_sequence = my_pfsa1.generate();
        let sequence_string = generated_sequence.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        println!("Generated Sequence {:15} with probability {1:.15}",sequence_string,my_pfsa1.prob_of_word(&generated_sequence));
        assert!(my_fsa1.accepts_sequence(&generated_sequence));
        assert!(my_fsa1_manual.accepts_sequence(&generated_sequence));
        assert!(!my_fsa2.accepts_sequence(&generated_sequence));
    }

    println!("\nOutput from FSA 2\n");

    for _i in 0..20 {
        let generated_sequence = my_pfsa2.generate();
        let sequence_string = generated_sequence.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        println!("Generated Sequence {:15} with probability {1:.15}",sequence_string,my_pfsa2.prob_of_word(&generated_sequence));
        assert!(!my_fsa1.accepts_sequence(&generated_sequence));
        assert!(!my_fsa1_manual.accepts_sequence(&generated_sequence));
        assert!(my_fsa2.accepts_sequence(&generated_sequence));
    }

}

fn cfg_example() {
    let my_cfg1 = ContextFreeGrammar::init_from_file("exampleCFG1.txt".to_string());
    let my_cfg2 = ContextFreeGrammar::init_from_file("exampleCFG2.txt".to_string());

    println!("\nOutput from CFG 1\n");

    for _i in 0..20 {
        println!("Generated: {}", my_cfg1.generate().join(""));
    }

    println!("\nOutput from CFG 2\n");

    for _i in 0..20 {
        println!("Generated: {}", my_cfg2.generate().join(""));
    }
}

fn ngram_example() {

    let my_corpus = vec![vec![1,2,3,4,5],vec![5,4,3,2,1]];
    let number_1gram = NGramModel::init(&my_corpus,1);
    let number_2gram = NGramModel::init(&my_corpus,2);
    let number_3gram = NGramModel::init(&my_corpus,3);
    let corpus3_2gram = NGramModel::init(&ngrams::open_as_words("exampleCorpus3.txt".to_string()),2);
    let corpus1_2gram = NGramModel::init(&ngrams::open_as_lines("exampleCorpus1.txt".to_string()),2);
    //println!("{:?}", ngrams::open_as_lines("exampleCorpus1.txt".to_string()));
    //println!("{:?}", ngrams::open_as_sentences("exampleCorpus2.txt".to_string()));
    //println!("{:?}", ngrams::open_as_words("exampleCorpus3.txt".to_string()));

    println!("\nOutput from number ngram 1\n");

    for _i in 0..10 {
        println!("Generated: {}", number_1gram.generate().iter().map(|s| s.to_string()).collect::<String>());
    }

    println!("\nOutput from number ngram 2\n");

    for _i in 0..10 {
        println!("Generated: {}", number_2gram.generate().iter().map(|s| s.to_string()).collect::<String>());
    }

    println!("\nOutput from number ngram 3\n");

    for _i in 0..5 {
        println!("Generated: {}", number_3gram.generate().iter().map(|s| s.to_string()).collect::<String>());
    }

    println!("\nOutput from Corpus 3 ngram\n");

    for _i in 0..5 {
        println!("Generated: {}", corpus3_2gram.generate().iter().map(|s| s.to_string()).collect::<String>());
    }

    println!("\nOutput from Corpus 1 ngram\n");

    for _i in 0..5 {
        println!("Generated: {}", corpus1_2gram.generate().join(" "));
    }

}

fn hmm_example() {

    let state_space: HashSet<String> = 
    ["A".to_string(),"B".to_string(),"C".to_string()].iter().cloned().collect();

    let observation_space: HashSet<String> = 
    ["a".to_string(),"b".to_string(),"c".to_string()].iter().cloned().collect();


    let initial_probabilities: HashMap<String, f64> =
    [("A".to_string(), 0.4),
     ("B".to_string(), 0.4),
     ("C".to_string(), 0.2),]
     .iter().cloned().collect();

    let transition_matrix: HashMap<(String,String),f64> =
    [(("A".to_string(),"A".to_string()), 0.6),
     (("A".to_string(),"B".to_string()), 0.3),
     (("A".to_string(),"C".to_string()), 0.1),
     (("B".to_string(),"A".to_string()), 0.3),
     (("B".to_string(),"B".to_string()), 0.6),
     (("B".to_string(),"C".to_string()), 0.1),
     (("C".to_string(),"A".to_string()), 0.1),
     (("C".to_string(),"B".to_string()), 0.1),
     (("C".to_string(),"C".to_string()), 0.8),]
     .iter().cloned().collect();

    let emission_matrix: HashMap<(String,String),f64> =
    [(("A".to_string(),"a".to_string()), 0.7),
     (("A".to_string(),"b".to_string()), 0.1),
     (("A".to_string(),"c".to_string()), 0.2),
     (("B".to_string(),"a".to_string()), 0.1),
     (("B".to_string(),"b".to_string()), 0.7),
     (("B".to_string(),"c".to_string()), 0.2),
     (("C".to_string(),"a".to_string()), 0.33),
     (("C".to_string(),"b".to_string()), 0.33),
     (("C".to_string(),"c".to_string()), 0.34),]
     .iter().cloned().collect();

    let my_hmm1 = HiddenMarkovModel::init(state_space,observation_space,initial_probabilities,transition_matrix,emission_matrix);

    //let manuel_state_sequence: Vec<String> = ["A".to_string(),"B".to_string()].iter().cloned().collect();
    //println!("{}",my_hmm1.prob_of_state_sequence(man_state_sequence));

    println!("\nState sequences from HMM1\n");

    for _i in 0..10 {
        let state_sequence = my_hmm1.generate_state_sequence(7);
        let sequence_string = state_sequence.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        println!("The state Sequence {:7} has probability {:.10} of being generated",sequence_string,my_hmm1.prob_of_state_sequence(state_sequence));
    }

    println!("\nEmission sequences from HMM1\n");

    for _i in 0..10 {
        let emission_sequence = my_hmm1.generate_emission_sequence(7);
        let sequence_string = emission_sequence.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        println!("The emission Sequence {:7} has probability {:.10} of being generated",sequence_string,my_hmm1.prob_of_observation_sequence(emission_sequence));
    }

    println!("\nViterbi on emission sequences from HMM1\n");

    for _i in 0..5 {
        let emission_sequence = my_hmm1.generate_emission_sequence(6);
        let viterbi_outcome = my_hmm1.viterbi(emission_sequence.clone());
        let sequence_string = emission_sequence.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        let state_string = viterbi_outcome.0.iter().map(|s| s.to_string()).collect::<Vec<String>>().join("");
        println!("Emission Seq {:6} was likely made by {:6} with probability of this state-emission sequence {:.15}",sequence_string,state_string,viterbi_outcome.1);
    }

}