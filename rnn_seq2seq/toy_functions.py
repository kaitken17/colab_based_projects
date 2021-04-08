import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

import numpy as np

import time

def generateInputWordBank(toy_params):
    """ Creates the word bank based on various inputs.
    words contains words that can be added to phrase
    all_words contains all possible words including special character words """
    base_words = toy_params['base_words']
    words = ['I'+str(i) for i in range(base_words)]

    all_words = words.copy()

    if toy_params['var_length']:
        all_words.append('<pad>')
    
    all_words_pp = all_words.copy()
    if toy_params['pad_dim']: 
        # Adds additional words that aren't ever used to input to increase effective embedding dimension
        if len(all_words) < toy_params['emb_dim']:
            all_words.extend(['<in_dim_pad>' for _ in range(toy_params['emb_dim'] - len(all_words))])

    return all_words, all_words_pp, words

def make_toy_phrase(toy_params):
    """ Creates a single toy phrase from the word bank """
    
    if toy_params['var_length']:
        phrase_length = np.random.randint(toy_params['min_phrase_len'], toy_params['phrase_length'])
    else:
        phrase_length = toy_params['phrase_length']
    in_words = toy_params['phrase_words'] # only generates phrases from words that can be in a phrase
    phrase = []
    n_words = len(in_words)

    for idx in range(phrase_length):
        next_word = False
        while not next_word: # This infrastructure saved for implementing rules later
            next_word = False
            word_idx = np.random.randint(n_words)
            next_word = True
        
        phrase.append(in_words[word_idx])

    if toy_params['var_length']: # Adds padding if variable length
        phrase.extend(['<pad>' for _ in range(toy_params['phrase_length'] - phrase_length)])

    return phrase

def generateOutputWordBank(toy_params):
    """ Generates the output word bank and translation dictionary which is used to translate phrases """
    base_words = toy_params['base_words']
    if 'rules' not in toy_params: toy_params['rules'] = [] # Defaults

    output_words = []
    base_word_vals = {}
    for i in range(base_words):
        base_word_vals['I'+str(i)] = 'O'+str(i)
        output_words.append('O'+str(i))
    if 'prev0' in toy_params['rules']:
        output_words.append('O0P')

    if toy_params['var_length']: 
        output_words.append('<pad>')
        base_word_vals['<pad>'] = '<pad>' # amount of padding in input = amount in output

    output_words_pp = output_words.copy()
    if toy_params['pad_dim']: 
        # Adds additional words that aren't ever used to output to increase effective embedding dimension
        if len(output_words) < toy_params['emb_dim']:
            output_words.extend(['<out_dim_pad>' for _ in range(toy_params['emb_dim'] - len(output_words))])

    return output_words, output_words_pp, base_word_vals

def translate_toy_phrase(toy_phrase, toy_params):
    """ Evaluates a single toy phrase and returns another phrase """
    base_words = toy_params['base_words']
    translations = toy_params['translations']
    phrase_length = len(toy_phrase)

    out_phrase = []

    for idx in range(phrase_length):
        word_found = False
        # Various rule checks
        if 'prev0' in toy_params['rules'] and idx > 1:
            if toy_phrase[idx] == 'I0' and toy_phrase[idx-1] == 'I0':
                word_found = True
                out_phrase.append('O0P')
        
        # Regular translation
        if not word_found and toy_phrase[idx] in list(translations.keys()):
            out_phrase.append(translations[toy_phrase[idx]])

    return out_phrase

def wordToIndex(word, word_bank):
    """ Converts a word into corresponding index in words """
    return word_bank.index(word)

def wordToTensor(word, word_bank):
    """ Turn a letter into a <1 x n_words> Tensor """
    n_words = len(word_bank)
    tensor = np.zeros((1, n_words))
    tensor[0][wordToIndex(word, word_bank)] = 1
    return np.array(tensor)

def eosTensor(word_bank):
    """ Returns <1 x n_words> EoS Tensor """
    n_words = len(word_bank)
    tensor = np.zeros((1, n_words))
    return np.array(tensor)

def phraseToTensor(phrase, word_bank):
    """ Turn a phrase into a <phrase_length x n_words> (an array of one-hot letter vectors) 
    Works for both inputs phrases and output phrases """
    n_words = len(word_bank)
    tensor = np.zeros((len(phrase), n_words))
    for wi, word in enumerate(phrase):
            tensor[wi][wordToIndex(word, word_bank)] = 1
    return np.array(tensor)

def tensorToPhrase(tensor, word_bank):
    """ Turn an array of one-hot letter vectors into a phrase """
    phrase = []
    for idx in range(tensor.shape[0]):
            hot_idx = np.argmax(tensor[idx])
            phrase.append(word_bank[hot_idx])
    return phrase

def randomTrainingExample(toy_params):
    """
    Generates a random training example consisting of nput phrase and sentiment and corresponding tensors

    Returns:
    sentiment_tensor: time x input dim (word bank size)
    sentiment_tensor: 1 x output dim (sentiment bank size)
    target_mask:

    """
    if 'reverse_out' not in toy_params: toy_params['reverse_out'] = False # Default

    # Unpacks toy_params (with defaults)
    phrase_length = toy_params['phrase_length']
    in_words = toy_params['in_words']
    out_words = toy_params['out_words']
    loss_type = toy_params['loss_type'] if 'loss_type' in toy_params else 'XE'

    in_phrase = make_toy_phrase(toy_params)
    in_phrase_tensor = phraseToTensor(in_phrase, toy_params['in_words'])
    out_phrase = translate_toy_phrase(in_phrase, toy_params)
    if toy_params['reverse_out']:
        if toy_params['var_length']: # Variable length reversal
            pad_idx = out_phrase.index('<pad>')
            out_phrase_pp = out_phrase[:pad_idx]
            out_phrase_pp.reverse()
            out_phrase = out_phrase_pp + out_phrase[pad_idx:]
        else: # Fixed length reversal
            out_phrase.reverse()
    out_phrase_tensor = phraseToTensor(out_phrase, toy_params['out_words'])

    # Determines phrase length (+1 for EoS character, -1 because index)
    in_phrase_length = in_phrase.index('<pad>') if toy_params['var_length'] else toy_params['phrase_length']-1
    out_phrase_length = out_phrase.index('<pad>') if toy_params['var_length'] else toy_params['phrase_length']-1
    in_target_mask = np.array(in_phrase_length, dtype=int) # When target is defined.
    out_target_mask = np.array(in_phrase_length, dtype=int) # When target is defined.
    
    return out_phrase, in_phrase, out_phrase_tensor, in_phrase_tensor, out_target_mask, in_target_mask