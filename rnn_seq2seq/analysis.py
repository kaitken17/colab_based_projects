import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from seq2seq_files import toy_functions as toy
from seq2seq_files import train_and_build as train_build


import sys
sys.path.append('/content/drive/My Drive/ml_research/fixedpoints_nlp/reverse-engineering-neural-networks')
import renn

def build_cell(rnn_specs):
    if 'cell_type' not in rnn_specs: rnn_specs['cell_type']  = 'GRU' # Default
    
    hm = 1 # hidden state size modifier
    if rnn_specs['arch'] == 'enc_dec_short':
        hm = 2

    print('Using {} cell'.format(rnn_specs['cell_type']))
    if rnn_specs['cell_type'] == 'Vanilla':
        cell = renn.VanillaRNN(hm * rnn_specs['hidden_size'])
    elif rnn_specs['cell_type'] == 'GRU':
        cell = renn.GRU(hm * rnn_specs['hidden_size'])
    elif rnn_specs['cell_type'] == 'LSTM':
        cell = renn.LSTM(hm * rnn_specs['hidden_size'])
    elif rnn_specs['cell_type'] == 'UGRNN':
        cell = renn.UGRNN(hm * rnn_specs['hidden_size'])
    
    if rnn_specs['ro_bias']:
        readout_init, readout_apply = stax.Dense(rnn_specs['output_size'])
    else:
        readout_init, readout_apply = train_build.Dense_nobias(rnn_specs['output_size'])

    if rnn_specs['arch'] in ('enc_dec_atth'):
        attention_init, attention_apply = train_build.dot_attention(rnn_specs['hidden_size'])
    elif rnn_specs['arch'] in ('enc_dec_attl'):
        attention_init, attention_apply = train_build.qkv_attention(rnn_specs['v_size'], rnn_specs['qk_size'])
    elif rnn_specs['arch'] in ('enc_dec_attmh'):
        attention_apply = []
        for _ in range(rnn_specs['attn_heads']):
                _, attention_apply_head = train_build.qkv_attention(rnn_specs['v_size'], rnn_specs['qk_size'])
                attention_apply.append(attention_apply_head)
        if rnn_specs['ro_bias']:
            _, context_out_apply = stax.Dense(rnn_specs['output_size'])
        else:
            _, context_out_apply = train_build.Dense_nobias(rnn_specs['output_size'])
        attention_apply.append(context_out_apply)
        attention_apply = tuple(attention_apply) # Converts to tuple so its hashable for JAX

    return cell, attention_apply, readout_apply

def build_applies(params, rnn_specs):
    """ Builds all the individual apply functions WITH the final_parameters """
    cell, attn_apply, readout_apply = build_cell(rnn_specs)

    if rnn_specs['arch'] in ('enc_dec_atth',):
        rnn_params_enc, rnn_params_dec, readout_params = params
    elif rnn_specs['arch'] in ('enc_dec_attl',):
        rnn_params_enc, rnn_params_dec, att_params, readout_params = params
        attn_apply = functools.partial(attn_apply, att_params)
    elif rnn_specs['arch'] in ('enc_dec_attmh',): # Architectures with learnable attention
        rnn_params_enc, rnn_params_dec, att_params, readout_params = params
        head_params, context_out_params = att_params
        attn_applys = []
        for idx, apply_fn in enumerate(attn_apply):
            if idx == len(attn_apply)-1: # last index is context_out
                attn_applys.append(functools.partial(apply_fn, context_out_params))
            else:
                attn_applys.append(functools.partial(apply_fn, head_params[idx]))
        attn_apply = attn_applys

        @jax.jit
        def logits_from_attn_weights(enc_seq, attn_rows):
            """
            enc_seq.shape = (batch, enc_len, hidden_dim)
            attn_rows.shape = (batch, attn_heads, enc_len)

            """
            c_vecs = []
            for head_idx, head_param in zip(range(rnn_specs['attn_heads']), head_params):
                _, _, v_mat = head_param
                values = jnp.matmul(enc_seq, v_mat)
                # (batch, seq_len) x (batch, seq_len, value_dim) -> (batch, value_dim)
                context_vec = jnp.einsum('ij,ijk->ik', attn_rows[:, head_idx], values)
                c_vecs.append(context_vec)

            context_total = jnp.concatenate(c_vecs, axis=1)
            context_vector = attn_apply[-1](context_total)

            return ro_apply(context_vector)

        
    enc_apply = functools.partial(cell.batch_apply, rnn_params_enc)
    dec_apply = functools.partial(cell.batch_apply, rnn_params_dec)
    ro_apply = functools.partial(readout_apply, readout_params)
    rnn_apply = enc_apply, dec_apply

    if rnn_specs['arch'] in ('enc_dec_attmh',): 
        return rnn_apply, attn_apply, ro_apply, logits_from_attn_weights
    else:
        return rnn_apply, attn_apply, ro_apply 

# Some helper functions to collect RNN hidden states
def _get_all_states(cells, attn_apply, ro_apply, inputs, in_masks, params, rnn_specs, return_hidden=True):
    def _get_all_states_jit(cells, attn_apply, ro_apply, inputs, in_masks, params):
        """Get RNN states in response to a batch of inputs (also returns masked states)."""
        print('Get all states with arch:', rnn_specs['arch'])
        
        if rnn_specs['arch'] == 'enc_dec': # Vanilla encoder decoder
            cell = cells 
            rnn_params_enc, rnn_params_dec, readout_params = params
            
            initial_states = cell.get_initial_state(rnn_params_enc, batch_size=inputs.shape[0])
            enc_apply = functools.partial(cell.batch_apply, rnn_params_enc)
            dec_apply = functools.partial(cell_dec.batch_apply, rnn_params_dec)
            rnn_apply = enc_apply, dec_apply
            
            return encode_decode(initial_states, inputs, rnn_apply, functools.partial(readout_apply, readout_params), 
                                                    return_hidden=return_hidden)
        elif rnn_specs['arch'] in ['enc_dec_att', 'enc_dec_short']: # Encoder decoder with attention/shorts
            cell, cell_dec = cells # Unpacks cells (encoder, decoder)
            
            def attention_apply(params, inputs):
                att_mat = params
                return jnp.dot(inputs, att_mat)
            
            if rnn_specs['arch'] == 'enc_dec_short':
                rnn_params_enc, rnn_params_dec, readout_params = params
                # Builds attend matrix based on shorts
                att_params = jnp.diag(jnp.asarray([int(i in rnn_specs['shorts']) for i in range(toy_params['phrase_length'])]))
            elif rnn_specs['arch'] == 'enc_dec_att':
                rnn_params_enc, rnn_params_dec, att_params, readout_params = params
            
            initial_states = cell.get_initial_state(rnn_params_enc, batch_size=inputs.shape[0])
            enc_apply = functools.partial(cell.batch_apply, rnn_params_enc)
            dec_apply = functools.partial(cell_dec.batch_apply, rnn_params_dec)
            rnn_apply = enc_apply, dec_apply
            att_apply = functools.partial(attention_apply, jax.nn.softmax(att_params, axis=0))
                
            return train_build.encode_decode_attn(initial_states, inputs, rnn_apply, att_apply, functools.partial(ro_apply, readout_params), 
                                                            return_hidden=return_hidden, zero_context=rnn_specs['zero_context'])
        
        elif rnn_specs['arch'] in ('enc_dec_atth', 'enc_dec_attl', 'enc_dec_attmh'): # Encoder decoder with attention via hidden states
            cell, cell_dec = cells # Unpacks cells (encoder, decoder)
            
            if rnn_specs['arch'] in ('enc_dec_atth',):
                rnn_params_enc, rnn_params_dec, readout_params = params
            elif rnn_specs['arch'] in ('enc_dec_attl',):
                rnn_params_enc, rnn_params_dec, att_params, readout_params = params
                attn_apply = functools.partial(attn_apply, att_params)
            elif rnn_specs['arch'] in ('enc_dec_attmh',): # Architectures with learnable attention
                rnn_params_enc, rnn_params_dec, att_params, readout_params = params
                head_params, context_out_params = att_params
                attn_applys = []
                for idx, apply_fn in enumerate(attn_apply):
                    if idx == len(attn_apply)-1: # last index is context_out
                        attn_applys.append(functools.partial(apply_fn, context_out_params))
                    else:
                        attn_applys.append(functools.partial(apply_fn, head_params[idx]))
                attn_apply = attn_applys
                
            initial_states = cell.get_initial_state(rnn_params_enc, batch_size=inputs.shape[0])
            enc_apply = functools.partial(cell.batch_apply, rnn_params_enc)
            dec_apply = functools.partial(cell_dec.batch_apply, rnn_params_dec)
            rnn_apply = enc_apply, dec_apply

            dec_seq_size = (inputs.shape[0], rnn_specs['out_len'], rnn_specs['output_size']) # (batch size, phrase len, output word size)
                        
            return train_build.encode_decode_attnh(initial_states, inputs, in_masks, rnn_apply, attn_apply, 
                                                   functools.partial(ro_apply, readout_params), return_hidden=return_hidden, 
                                                   rnn_specs=rnn_specs, dec_seq_size=dec_seq_size)
    
    return jax.jit(_get_all_states_jit, static_argnums=(0, 1, 2))(cells, attn_apply, ro_apply, inputs, in_masks, params)

def rnn_states(cells, attn_apply, ro_apply, batch, params, rnn_specs, return_hidden=True):
    """Return RNN states.

    return_hidden: determines if apply returns hidden states (True) or logits (False)
    """

    logits, states = _get_all_states(cells, attn_apply, ro_apply, batch['inputs'], batch['in_index'], params, rnn_specs, return_hidden=return_hidden)

    if rnn_specs['cell_type'] != 'LSTM': # Keeping all of the hidden state for non-LSTM
        return logits, [h for h in states]
    else: # For LSTM, we only send the first half of the "hidden state" to the readout, because this corresponds to the true hidden state (i.e. not the cell part)
        hidden_state_keep = int(1/2*states[0].shape[-1])
        return logits, [h[:, :hidden_state_keep] for h in states]

def keep_unique_fixed_points(fps, identical_tol=0.0, do_print=True):
    """Get unique fixed points by choosing a representative within tolerance.
    Args:
        fps: numpy array, FxN tensor of F fixed points of N dimension
        identical_tol: float, tolerance for determination of identical fixed points
        do_print: Print useful information? 
    Returns:
        2-tuple of UxN numpy array of U unique fixed points and the kept indices
    """
    keep_idxs = np.arange(fps.shape[0])
    if identical_tol <= 0.0:
        return fps, keep_idxs
    if fps.shape[0] <= 1:
        return fps, keep_idxs
    
    nfps = fps.shape[0]
    example_idxs = np.arange(nfps)
    all_drop_idxs = []

    # If point a and point b are within identical_tol of each other, and the
    # a is first in the list, we keep a.
    distances = squareform(pdist(fps, metric="euclidean"))
    for fidx in range(nfps-1):
        distances_f = distances[fidx, fidx+1:]
        drop_idxs = example_idxs[fidx+1:][distances_f <= identical_tol]
        all_drop_idxs += list(drop_idxs)
             
    unique_dropidxs = np.unique(all_drop_idxs)
    keep_idxs = np.setdiff1d(example_idxs, unique_dropidxs)
    if keep_idxs.shape[0] > 0:
        unique_fps = fps[keep_idxs, :]
    else:
        unique_fps = np.array([], dtype=np.int64)

    if do_print:
        print("    Kept %d/%d unique fixed points with uniqueness tolerance %f." %
                    (unique_fps.shape[0], nfps, identical_tol))
        
    return unique_fps, keep_idxs

def participation_ratio_vector(C):
    """Computes the participation ratio of a vector of variances."""
    return np.sum(C) ** 2 / np.sum(C*C)

def generate_labels(toy_params):
    """ Generates all possible labels a phrase can have """
    total_labels = len(toy_params['out_words'])**toy_params['phrase_length']
    labels = []
    for i in range(toy_params['phrase_length']):
        if labels == []:
            labels = [[i] for i in range(len(toy_params['out_words']))]
        else:
            new_labels = []
            for j in range(len(labels)):
                for k in range(len(toy_params['out_words'])):
                    temp_label = labels[j].copy()
                    temp_label.append(k)
                    new_labels.append(temp_label)
            labels = new_labels

    return labels


def subspace_perc(A, b):
    """
    A: an N x M matrix, whose column space represents some M-dimensional subspace
    b: an N x P vector
    For each of b's columns, finds the percentage of its magnitude which lies in 
    the subspace formed by the columns of A (its columnspace). Then averages these together
    """
    proj = np.matmul(np.matmul(A, np.linalg.inv(np.matmul(A.T, A))), A.T) # N x N matrix
    b_proj = np.matmul(proj, b) # N x P martix

    norm_perc = np.linalg.norm(b_proj, axis=0)/np.linalg.norm(b, axis=0) # P dim vector

    return np.mean(norm_perc)

def ro_subspace_analysis(readout):
    readout = np.asarray(readout)
    ro_n = readout.shape[0]
    indices = [i for i in range(ro_n)]
    for ss_dim in range(1, ro_n): # Tries dimensions from 1 to ro_n - 1
        comb = combinations(indices, ss_dim) # Gets all possible combinations
        n_comb = 0
        perc_vals = 0
        for subspace_idxs in list(comb): # For each combination
            n_comb += 1
            A_idxs = list(subspace_idxs)
            b_idxs = list(filter(lambda a: a not in list(subspace_idxs), indices))
            A = readout[A_idxs]
            b = readout[b_idxs] 
            perc_vals += subspace_perc(A.T, b.T)
        print('Avg perc in rest for subspace dim {}:'.format(ss_dim), 1/n_comb * perc_vals)


def find_avg_hs(hs, hs_data, toy_params):
    """ Finds the average hidden states, accounting for pads """

    print('Finding average hidden state at each time step...')

    enc_len = hs_data['inputs'][0].shape[0]
    dec_len = hs_data['labels'][0].shape[0]

    avg_hs = np.zeros((hs['all'][0].shape))
    if toy_params['var_length']:
        pad_in_idx = toy_params['in_words'].index('<pad>')
        pad_out_idx = toy_params['out_words'].index('<pad>')

    for time_idx in range(hs['all'][0].shape[0]):
        if time_idx < enc_len: # Encoder
            time_hs = []
            for h, inp in zip(hs['all'], hs_data['inputs']):
                if time_idx > 0 and toy_params['var_length']:
                    if np.argmax(inp[time_idx]) != pad_in_idx or np.argmax(inp[time_idx-1]) != pad_in_idx: # Does not add repeated <pads>
                        time_hs.append(h[time_idx])
                else: # Automatically adds if first index
                    time_hs.append(h[time_idx])
            time_hs = np.asarray(time_hs)
        else: # Decoder
            time_hs = []
            for h, out in zip(hs['all'], hs_data['labels']):
                if time_idx - enc_len> 0 and toy_params['var_length']:
                    if np.argmax(out[time_idx-enc_len]) != pad_out_idx or np.argmax(out[time_idx-enc_len-1]) != pad_out_idx:
                        time_hs.append(h[time_idx])
                else: # Automatically adds if first index
                    time_hs.append(h[time_idx])
            time_hs = np.asarray(time_hs)
        if time_hs.shape[0] > 0:
            avg_hs[time_idx] = np.mean(time_hs, axis=0)
        else:
            print('Zero length')

    return avg_hs

def sort_hs_by_time_and_word(hs, hs_data, toy_params, subtract_avg=False, avg_hs=0.0):
    """ Returns two nested lists of hidden states, one for the encoder and decoder.
    They are each indexed as [time_idx][word_idx]. Can optionally subtract average
    hidden state from all hidden states. """

    enc_len = hs_data['inputs'][0].shape[0]
    dec_len = hs_data['labels'][0].shape[0]
    
    if subtract_avg:
        avg_hs = find_avg_hs(hs, hs_data, toy_params)
        hidden_states = [h - avg_hs  for h in hs['all']]
    else:
        hidden_states = hs['all']

    hs_time_word_enc = [[[] for _ in range(len(toy_params['in_words_pp']))] for _ in range(enc_len)]
    # + 1 because will be looking for <BoS> as well, which is not included in `out_words_pp`
    hs_time_word_dec = [[[] for _ in range(len(toy_params['out_words_pp']) + 1)] for _ in range(dec_len)]

    pad_in_idx = toy_params['in_words'].index('<pad>')
    pad_out_idx = toy_params['out_words'].index('<pad>') 

    for time_idx in range(enc_len): # Encoder
        # time_hs_word = [[] for _ in range(len(toy_params['in_words_pp']))]
        for h, inp in zip(hidden_states, hs_data['inputs']):
            if time_idx > 0:
                if np.argmax(inp[time_idx]) != pad_in_idx or np.argmax(inp[time_idx-1]) != pad_in_idx: # Does not add repeated <pads>
                    hs_time_word_enc[time_idx][np.argmax(inp[time_idx])].append(h[time_idx])
            else: # Automatically adds if first index
                hs_time_word_enc[time_idx][np.argmax(inp[time_idx])].append(h[time_idx])
    for time_idx in range(dec_len): # Decoder
        # time_hs_word = [[] for _ in range(len(toy_params['out_words_pp']))]
        for h, dec_inp_idx in zip(hidden_states, hs_data['dec_inputs']):
            if time_idx > 0:
                if dec_inp_idx[time_idx] != pad_out_idx or dec_inp_idx[time_idx-1] != pad_out_idx:
                    hs_time_word_dec[time_idx][dec_inp_idx[time_idx]].append(h[enc_len + time_idx])
            else: # Automatically adds if first index
                hs_time_word_dec[time_idx][dec_inp_idx[time_idx]].append(h[enc_len + time_idx])

    return hs_time_word_enc, hs_time_word_dec

def angle_degrees(a, b):
    return 180/np.pi * np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def proj_perc(a, b):
    """ Projects a onto b and then sees what percentage of magnitude lives in space """
    norm_a = np.linalg.norm(a)
    norm_proj_a = np.linalg.norm(a - np.dot(a, b) * (b / np.linalg.norm(b)))
    return norm_proj_a/norm_a

def find_scan_offset(hs_data, toy_params):
    """ Finds the average offset of each index in SCAN """
    enc_len = hs_data['inputs'][0].shape[0]
    dec_len = hs_data['labels'][0].shape[0]

    displacements = np.zeros((len(hs_data['inputs']), hs_data['inputs'][0].shape[0]))
    length_count = np.zeros((enc_len))
    and_idx = toy_params['in_words'].index('and')
    left_idx = toy_params['in_words'].index('left')

    for idx, h_inp, in_idx in zip(range(len(hs_data['inputs'])), hs_data['inputs'], hs_data['in_index']):
        length_count[:in_idx] += np.ones((in_idx)) # Tracks length of phrases

        and_locs = np.argmax(h_inp, axis=1) == and_idx
        left_locs = np.argmax(h_inp, axis=1) == left_idx

        and_locs = np.asarray(and_locs, dtype=np.int32)
        left_locs = -1 * np.asarray(left_locs, dtype=np.int32) # Converts into integers
        left_locs = left_locs -1 * np.roll(left_locs, shift=-1) # Accounts for lefts and word after left

        displacements[idx, :in_idx] = [np.sum(and_locs[:i+1] + left_locs[:i+1]) for i in range(in_idx)]

    # Does a weighted average based on lengths
    displacements = np.sum(displacements, axis=0)
    avg_offset = displacements/length_count

    return avg_offset

def calculate_input_components(hs, hs_data, toy_params, rnn_specs):
    """ Calculates the input components when given collection of hidden states with average already subtracted"""
    enc_len = hs_data['inputs'][0].shape[0]
    dec_len = hs_data['labels'][0].shape[0]
    
    hs_time_word_enc_ic, hs_time_word_dec_ic = sort_hs_by_time_and_word(hs, hs_data, toy_params, subtract_avg=True)
    
    ic_enc = np.zeros((len(toy_params['in_words_pp']), rnn_specs['hidden_size']))

    for word_idx in range(len(toy_params['in_words_pp'])):
        word_hs = []
        for time_idx in range(enc_len):
            word_hs.extend(hs_time_word_enc_ic[time_idx][word_idx])
        ic_enc[word_idx] = np.mean(np.array(word_hs), axis=0)

    # Again, +1 is for BoS
    ic_dec = np.zeros((len(toy_params['out_words_pp']) + 1, rnn_specs['hidden_size']))

    for word_idx in range(len(toy_params['out_words_pp'])+1):
        word_hs = []
        for time_idx in range(dec_len):
            word_hs.extend(hs_time_word_dec_ic[time_idx][word_idx])
        ic_dec[word_idx] = np.mean(np.array(word_hs), axis=0)

    return ic_enc, ic_dec

def get_attn_align(rnn_specs, params, head=0):
    """ Alignment function for different architectures """
    def align_vals(enc, dec, k_mat=0, q_mat=0):
        k_enc = np.matmul(enc, k_mat)
        q_dec = np.matmul(dec, q_mat)
        return np.matmul(k_enc, q_dec.T)

    if rnn_specs['arch'] in ('enc_dec_attl'):
        _, _, att_params, _ = params
        q_mat, k_mat, v_mat = att_params

        attn_align = functools.partial(align_vals, k_mat=k_mat, q_mat=q_mat)
    elif rnn_specs['arch'] in ('enc_dec_attmh'):
        _, _, att_params, _ = params
        # Only returns align for a single head
        q_mat, k_mat, v_mat = att_params[0][head]
        attn_align = functools.partial(align_vals, k_mat=k_mat, q_mat=q_mat)
    else:
        attn_align = np.dot

    return attn_align

def find_align_breakdown(hs, hs_data, attn_align, toy_params, rnn_specs):

    temporal_hs = find_avg_hs(hs, hs_data, toy_params)
    ic_enc, ic_dec = calculate_input_components(hs, hs_data, toy_params, rnn_specs)
    # print('Using null input hidden state as temporal part')
    # temporal_hs = hs['special'][-1]

    enc_len = hs_data['inputs'][0].shape[0]
    dec_len =  hs_data['labels'][0].shape[0]

    dot_keys = ['full_full', 'temp_temp', 'temp_inp', 'temp_del', 'inp_temp', 'inp_inp', 'inp_del', 'del_temp', 'del_inp', 'del_del']

    align_vals = {}
    for dot_key in dot_keys:
        align_vals[dot_key] = -np.inf * np.ones((hs_data['inputs'].shape[0], enc_len, dec_len))

    for h, h_idx, inp, out, in_idx, out_idx in zip(hs['all'], range(len(hs['all'])), hs_data['inputs'], hs_data['dec_inputs'], hs_data['in_index'], hs_data['out_index']):
        enc_seq = h[:enc_len]
        enc_temp = temporal_hs[:enc_len]
        # Builds encoder hs phrase from input components
        enc_phrase_input_idxs = np.argmax(inp, axis=1)
        enc_inp = np.zeros((enc_len, rnn_specs['hidden_size']))
        for enc_idx in range(enc_len):
            enc_inp[enc_idx] = ic_enc[enc_phrase_input_idxs[enc_idx]]
        enc_del = enc_seq - enc_temp - enc_inp

        for dec_idx in range(out_idx+1): # +1 to include out_idx in attention
            dec_h = h[enc_len + dec_idx]
            dec_temp = temporal_hs[enc_len + dec_idx]
            dec_inp = ic_dec[out[dec_idx]]
            dec_del = dec_h - dec_temp - dec_inp

            # Auto masks if beyond in_index length
            align_vals['full_full'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_seq, dec_h)[:in_idx+1]
            align_vals['temp_temp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_temp, dec_temp)[:in_idx+1]
            align_vals['temp_inp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_temp, dec_inp)[:in_idx+1]
            align_vals['temp_del'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_temp, dec_del)[:in_idx+1]
            align_vals['inp_temp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_inp, dec_temp)[:in_idx+1]
            align_vals['inp_inp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_inp, dec_inp)[:in_idx+1]
            align_vals['inp_del'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_inp, dec_del)[:in_idx+1]
            align_vals['del_temp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_del, dec_temp)[:in_idx+1]
            align_vals['del_inp'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_del, dec_inp)[:in_idx+1]
            align_vals['del_del'][h_idx, :in_idx+1, dec_idx] = attn_align(enc_del, dec_del)[:in_idx+1]

    return align_vals


def calculate_dot_ratios(align_values, num_aligns, hs_data):
    """ Calcualte dot ratios 
    num_aligns: Number of top alignments to take (1 = take only maximum value)
    
     """
    enc_len = hs_data['inputs'][0].shape[0]
    dec_len =  hs_data['labels'][0].shape[0]

    dot_keys = ['temp_temp', 'temp_inp', 'temp_del', 'inp_temp', 'inp_inp', 'inp_del', 'del_temp', 'del_inp', 'del_del']

    dts = align_values
    dot_ratios = {}
    for dot_key in dot_keys:
        dot_ratios[dot_key] = [[] for _ in range(dec_len)]

    for idx in range(dts['full_full'].shape[0]):
        for dec_idx in range(dec_len):
            if np.max(dts['full_full'][idx][:, dec_idx]) != -np.inf: # Doesn't look at rows with just infinities
                
                max_idxs = np.argpartition(dts['full_full'][idx, :, dec_idx], -num_aligns)[-num_aligns:] # Gets top indices
                max_idx_sort = dts['full_full'][idx, max_idxs, dec_idx].argsort()[::-1] # Sorts top indices by size
                max_idxs = max_idxs[max_idx_sort]
                
                max_idx = np.argmax(dts['full_full'][idx][:, dec_idx])
                total_sums = np.sum(np.array([np.abs(dts[dot_key][idx, max_idxs, dec_idx]) for dot_key in dot_keys]), axis=0)
                for dot_key in dot_keys:
                    dot_ratios[dot_key][dec_idx].append(np.abs(dts[dot_key][idx, max_idxs, dec_idx])/total_sums)

    dot_ratio_vals = np.zeros((len(dot_keys), dec_len))
    for dot_key, idx in zip(dot_keys, range(len(dot_keys))):
        for dec_idx in range(dec_len):
            # Averages over both num_aligns and all occurences
            dot_ratio_vals[idx, dec_idx] = np.mean(dot_ratios[dot_key][dec_idx])

    return dot_ratio_vals


def var_before_after_temp(hs, hs_data, toy_params):
    """ Calculates variance before and after temporal component has been subtracted out for encoder and decoder.
            Automatically removes <pad> if present in dataset. """
    enc_len = hs_data['inputs'][0].shape[0]
    dec_len = hs_data['labels'][0].shape[0]

    hs_time_word_enc, hs_time_word_dec = sort_hs_by_time_and_word(hs, hs_data, toy_params)
    hs_time_word_enc_nt, hs_time_word_dec_nt = sort_hs_by_time_and_word(hs, hs_data, toy_params, subtract_avg=True)
    avg_hs = find_avg_hs(hs, hs_data, toy_params)

    in_words = toy_params['in_words_pp'].copy()
    out_words = toy_params['out_words_pp'].copy()
    # Removes pads
    if '<pad>' in in_words: in_words.remove('<pad>')
    if '<pad>' in out_words: out_words.remove('<pad>')

    word_var_enc = np.zeros((len(in_words)))
    word_var_enc_nt = np.zeros((len(in_words)))
    word_var_dec = np.zeros((len(out_words)))
    word_var_dec_nt = np.zeros((len(out_words)))

    for word_idx in range(len(in_words)):
        word_hs = []
        word_hs_ic = []
        for time_idx in range(enc_len):
            word_hs.extend(hs_time_word_enc[time_idx][word_idx])
            word_hs_ic.extend(hs_time_word_enc_nt[time_idx][word_idx])

        word_hs = np.array(word_hs) 
        word_var_enc[word_idx] = np.trace(np.cov(word_hs.T))

        word_hs_ic = np.array(word_hs_ic) 
        word_var_enc_nt[word_idx] = np.trace(np.cov(word_hs_ic.T))

    for word_idx in range(len(out_words)):
        word_hs = []
        word_hs_ic = []
        for time_idx in range(dec_len):
            word_hs.extend(hs_time_word_dec[time_idx][word_idx])
            word_hs_ic.extend(hs_time_word_dec_nt[time_idx][word_idx])

        word_hs = np.array(word_hs) 
        word_var_dec[word_idx] = np.trace(np.cov(word_hs.T))

        word_hs_ic = np.array(word_hs_ic) 
        word_var_dec_nt[word_idx] = np.trace(np.cov(word_hs_ic.T))

    return word_var_enc, word_var_enc_nt, word_var_dec, word_var_dec_nt

def cross_entropy(logits, labels, lengths):
    
    def mask_sequences_one(sequences, last_index):
        """Set positions beyond the length of each sequence to 0. For sequences of shape (batch_size, seq_len)"""
        # sequences.shape = (batch_size, seq_length)
        # last_index.shape = (batch_size)
        mask = last_index[:, np.newaxis] >= np.arange(sequences.shape[1])
        return sequences * mask

    shifted = logits - logits.max(axis=-1, keepdims=True)
    # Log(e^x_i/sum_I e^x_I) = x_i - Log(sum_I e^x_I)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    labels_max = np.argmax(labels, axis=2)
    
    logliklihood = np.take_along_axis(log_probs, np.expand_dims(labels_max, axis=2), axis=2)
    
    # Masks the cross entropy loss
    xe_loss = -1 * mask_sequences_one(np.squeeze(logliklihood), lengths)

    return xe_loss

def logit_accuracy(logits, labels, out_indexs):
    """ Computes accuracy from logits """

    def mask_sequences(sequences, last_index):
        """Set positions beyond the length of each sequence to 0. For sequences of shape (batch_size, seq_len, vocab_size)"""
        mask = last_index[:, np.newaxis] >= np.arange(sequences.shape[1])
        return sequences * np.repeat(mask[:,:,np.newaxis], sequences.shape[-1], axis=2)

    mask_logits = mask_sequences(logits, out_indexs)
    mask_labels = mask_sequences(labels, out_indexs)
    
    predictions = np.argmax(mask_logits, axis=2).astype(jnp.int32)
    labels_max = np.argmax(mask_labels, axis=2).astype(jnp.int32)
    # Returns both accs for entire phrase and individual words
    return np.mean(np.all(labels_max == predictions, axis=1)), np.mean(labels_max == predictions)

def mask_std(array, axis=0, weights=None):
    """ A version of np.std that takes a mask """
    array_mean = np.average(array, axis=axis, weights=weights)
    x = (array - array_mean)**2
    return np.sqrt(np.average(x, axis=axis, weights=weights))