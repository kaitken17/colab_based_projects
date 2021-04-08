import os

import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

import numpy as np
import matplotlib.pyplot as plt

import time

from seq2seq_files import toy_functions as toy


def read_and_split_file(file_name: str) -> list:
    text_data = list()
    current_file = os.path.abspath(file_name)
    if os.path.exists(current_file):
        open_file = open(current_file, 'r', encoding="latin-1")
        text_data = open_file.read().split('\n')
        text_data = list(filter(None, text_data))
    return text_data

def read_and_filter_scan(text_data, input_words):
    """ Reads in phrases from text_data and filters phrases to keep only those that have words in input_words """
    in_phrases = []
    out_phrases = []

    for i in range(len(text_data)): 
        in_idx = text_data[i].find('IN: ') + 4
        out_idx = text_data[i].find('OUT: ') + 5
        input = text_data[i][in_idx:out_idx-6]
        output = text_data[i][out_idx:]
        input_phrase = input.split(' ')
        output_phrase = output.split(' ')

        if all([input_phrase[j] in input_words for j in range(len(input_phrase))]):
            in_phrases.append(input_phrase)
            out_phrases.append(output_phrase)
        
    return in_phrases, out_phrases

def pad_and_tensor_scan(in_phrases, out_phrases, scan_params):
    """ Pads a set of scan in/out phrases and converts to tensor outputs (either raw or extended sets) """

    if '<pad>' not in scan_params['in_words']:
        scan_params['in_words'].append('<pad>')
        scan_params['out_words'].append('<pad>')
    scan_params['in_words_pp'] = scan_params['in_words'].copy()
    scan_params['out_words_pp'] = scan_params['out_words'].copy()
    if scan_params['pad_dim']: 
        # Adds additional words that aren't ever used to input and output words to increase effective embedding dimension
        if len(scan_params['in_words']) < scan_params['emb_dim']:
            scan_params['in_words'].extend(['<in_dim_pad>' for _ in range(scan_params['emb_dim'] - len(scan_params['in_words']))])
        if len(scan_params['out_words']) < scan_params['emb_dim']:
            scan_params['out_words'].extend(['<out_dim_pad>' for _ in range(scan_params['emb_dim'] - len(scan_params['out_words']))])

    
    max_in = max(list(map(len, in_phrases))) + 1 # Plus 1 because adding a single <pad> to longest phrase
    max_out = max(list(map(len, out_phrases))) + 1

    scan_inputs_np = np.zeros((len(in_phrases), max_in, len(scan_params['in_words'])))
    scan_targets_np = np.zeros((len(out_phrases), max_out, len(scan_params['out_words'])))
    scan_target_masks_np = np.zeros((len(in_phrases),))
    scan_input_masks_np = np.zeros((len(in_phrases),))

    for i in range(len(in_phrases)):
        scan_input_masks_np[i] = len(in_phrases[i])
        scan_target_masks_np[i] = len(out_phrases[i])
        in_phrases[i].extend(['<pad>' for _ in range(max_in - len(in_phrases[i]))])
        out_phrases[i].extend(['<pad>' for _ in range(max_out - len(out_phrases[i]))])

        scan_inputs_np[i] = toy.phraseToTensor(in_phrases[i], scan_params['in_words'])
        scan_targets_np[i] = toy.phraseToTensor(out_phrases[i], scan_params['out_words'])

        # print('Input:', input_phrase)
        # print('Input tensor:', in_phrase_tensor)
        # print('Output:', output_phrase)
        # print('Output tensor:', out_phrase_tensor

    scan_data_np = {
        'inputs': scan_inputs_np,  # Phrase tensors: dataset_size x phrase_len x in_dim
        'labels': scan_targets_np, # Sentiment tensors: dataset_size x phrase_len x out_dim
        'in_index': scan_input_masks_np, # Target mask: phrase_len
        'out_index': scan_target_masks_np, # Target mask: phrase_len
    }

    return scan_data_np, scan_params

def extend_scan(raw_in_phrases, raw_out_phrases, scan_params, trials=1,  plot_lens=False):
    """ increases the length of SCAN phrases by appending them to one another"""
    def random_scan_phrase():
        """ Returns a random SCAN input/ouput phrase pair """
        if scan_params['multiply']: # nested lists, so uniform over TYPE of phrase
            rand_type = np.random.randint(len(raw_in_phrases))
            rand_idx = np.random.randint(len(raw_in_phrases[rand_type]))
            return raw_in_phrases[rand_type][rand_idx], raw_out_phrases[rand_type][rand_idx]
        else: # uniform over all phrases of phrase
            rand_idx = np.random.randint(len(raw_in_phrases))
            return raw_in_phrases[rand_idx], raw_out_phrases[rand_idx]


    min_len = scan_params['min_out_len']
    max_len = scan_params['max_out_len']
    
    new_ins = []
    new_outs = []

    if '<.>' not in scan_params['in_words'] and scan_params['periods']:
        scan_params['in_words'].append('<.>')
        scan_params['out_words'].append('<.>')

    while len(new_ins) < trials:
        new_in = []
        new_out = []
        while len(new_out) < min_len:

            rand_in_phrase, rand_out_phrase = random_scan_phrase()

            if len(new_out) + len(rand_out_phrase) + 1 > max_len:
                continue
            new_in.extend(rand_in_phrase)
            new_out.extend(rand_out_phrase)

            if scan_params['periods']:
                new_in.append('<.>')
                new_out.append('<.>')

        # Does not accept new phrase if greater than maximum input length (eliminates some weird outliers)
        if len(new_in) <= scan_params['max_in_len']:
            new_ins.append(new_in)
            new_outs.append(new_out)
        # else:
        #   print('Input too long: rejected')

    if plot_lens:
        in_lens = list(map(len, new_ins))
        out_lens = list(map(len, new_outs))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(in_lens)
        ax2.hist(out_lens)
        ax1.set_xlabel('Input lengths')
        ax2.set_xlabel('Output lengths')
    
    return new_ins, new_outs, scan_params

def generate_scan(raw_in_phrases, raw_out_phrases, scan_params, trials=1):
    """ Function used to generate scan phrases (used for extended and normal) dataset """
    start_time = time.time()
    if scan_params['extend_scan']: # extends scan phrases
        in_phrases, out_phrases, scan_params = extend_scan(raw_in_phrases, raw_out_phrases, scan_params, trials=trials, 
                                                                                                             plot_lens=scan_params['plot_lens'])
    else:
        in_phrases = raw_in_phrases
        out_phrases = raw_out_phrases

    # Pads a set of scan and converts to tensor outputs
    scan_data_np, scan_params = pad_and_tensor_scan(in_phrases, out_phrases, scan_params)

    scan_data = {
        'inputs': jnp.asarray(scan_data_np['inputs']),  # Phrase tensors: dataset_size x phrase_len x in_dim
        'labels': jnp.asarray(scan_data_np['labels']), # Sentiment tensors: dataset_size x phrase_len x out_dim
        'in_index': jnp.asarray(scan_data_np['in_index'], dtype=jnp.int32), # Target mask: phrase_len
        'out_index': jnp.asarray(scan_data_np['out_index'], dtype=jnp.int32), # Target mask: phrase_len
    }

    print('SCAN data generated in: {:0.2f} sec.'.format(time.time() - start_time))

    return scan_data, scan_params

def filter_out_left_twice(raw_in_phrases, raw_out_phrases, scan_params):
    print('Filtering out left twice occurences.')

    new_in_phrases = []
    new_out_phrases = []

    for phrase, out_phrase in zip(raw_in_phrases, raw_out_phrases):
        phrase_idxs = np.array([scan_params['in_words'].index(word) for word in phrase])

        left_locs = phrase_idxs == scan_params['in_words'].index('left')
        twice_locs = phrase_idxs == scan_params['in_words'].index('twice')

        if not any(np.logical_and(left_locs[:-1], twice_locs[1:])):
            new_in_phrases.append(phrase)
            new_out_phrases.append(out_phrase)

    print('Before phrases:', len(raw_in_phrases))
    print('After phrases:', len(new_in_phrases))

    return new_in_phrases, new_out_phrases