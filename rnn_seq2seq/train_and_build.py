import functools

import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

import numpy as np
import matplotlib.pyplot as plt

import os
import time
import pickle

from seq2seq_files import toy_functions as toy
from seq2seq_files import scan_functions as scan

import sys
sys.path.append('/content/drive/My Drive/ml_research/fixedpoints_nlp/reverse-engineering-neural-networks')
import renn

def build_rnn(rnn_specs):
    """Builds a single layer RNN (Vanilla/GRU/LSTM/UGRNN with readout)"""

    vocab_size = rnn_specs['input_size']

    if rnn_specs['arch'] in ['enc_dec_short', 'enc_dec_att']: # Attention mechanisms double hidden size
        num_units = 2*rnn_specs['hidden_size']
        print('Doubling hidden units for attention.')
    else:
        num_units = rnn_specs['hidden_size']
    num_outputs = rnn_specs['output_size']

    def build_cell_rnn(units):
        if rnn_specs['cell_type'] == 'Vanilla':
            cell = renn.rnn.cells.VanillaRNN(units)
        elif rnn_specs['cell_type'] == 'GRU':
            cell = renn.rnn.cells.GRU(units)
        elif rnn_specs['cell_type'] == 'LSTM':
            cell = renn.rnn.cells.LSTM(units)
        elif rnn_specs['cell_type'] == 'UGRNN':
            cell = renn.rnn.cells.UGRNN(units)
        return cell

    def build_readout(num_outputs):
        if rnn_specs['ro_bias']:
            readout_init, readout_apply = stax.Dense(num_outputs)
        else:
            readout_init, readout_apply = Dense_nobias(num_outputs)
        return readout_init, readout_apply

    def build_dot_attention(rnn_specs):
        attention_init, attention_apply = dot_attention(rnn_specs['hidden_size'])  
        return attention_init, attention_apply  

    def build_qkv_attention(rnn_specs):
        attention_init, attention_apply = qkv_attention(rnn_specs['v_size'], rnn_specs['qk_size'])  
        return attention_init, attention_apply  

    def build_attention_old():
        phrase_length = rnn_specs['phrase_length']
        attention_init, attention_apply = attention_old(phrase_length)  
        return attention_init, attention_apply  

    def init_fun(prng_key, enc_input_shape, dec_input_shape):
        """Initializes the network (the embedding and the RNN cell)."""
        k0, k1, k2, k3 = jax.random.split(prng_key, 4)
        output_shape, rnn_params = cell.init(k0, enc_input_shape[1:])
        _, rnn_params_dec = cell_dec.init(k1, dec_input_shape[1:])
        
        if rnn_specs['arch'] in ['enc_dec_short', 'enc_dec_att']: # Readout size reduced by 1/2 since only uses first half of hidden
            output_shape = output_shape[:-1] + (int(1/2*output_shape[-1]),)
        elif rnn_specs['arch'] in ('enc_dec_atth', 'enc_dec_attl', 'enc_dec_attmh'):
            # This code should be modified if context size is ever not equal to hidden size
            if rnn_specs['zero_rec']: # AO (output shape is unmodified)
                output_shape = output_shape[:-1] + (output_shape[-1],)
            elif rnn_specs['zero_attention']: #VED (output shape is unmodified)
                output_shape = output_shape[:-1] + (output_shape[-1],)
            else: #AED: readout size is a concatanation of hidden and context
                output_shape = output_shape[:-1] + (output_shape[-1] + output_shape[-1],)
        if rnn_specs['cell_type'] == 'LSTM': # Readout reduced by another factor of 1/2 since we are not using the cell part of the "hiden state"
            raise NotImplementedError('Need to update this code for LSTMs')
            output_shape = output_shape[:-1] + (int(1/2*output_shape[-1]),)

        # We explicitly add the batch dimension back in.
        # (This is because we only applied `vmap` to the rnn_apply function above)
        batch_output_shape = (enc_input_shape[0],) + output_shape
        _, readout_params = readout_init(k2, output_shape)

        if rnn_specs['arch'] in ('enc_dec_short', 'enc_dec_atth'):
            return batch_output_shape, (rnn_params, rnn_params_dec, readout_params)
        elif rnn_specs['arch'] in ('enc_dec_attl',):
            _, att_params = attention_init(k3, (rnn_specs['hidden_size'],))
            return batch_output_shape, (rnn_params, rnn_params_dec, att_params, readout_params)
        elif rnn_specs['arch'] in ('enc_dec_attmh',):
            ks = jax.random.split(k3, 1 + rnn_specs['attn_heads'])
            # Context has batch size built into its init just like readout
            batch_cout_input_shape = (enc_input_shape[0],) +(rnn_specs['attn_heads'] * rnn_specs['v_size'],)
            _, context_out_params = context_out_init(ks[0], batch_cout_input_shape)
            head_params = []
            for idx, head_init in enumerate(attention_inits):
                _, att_params = head_init(ks[1+idx], (rnn_specs['hidden_size'],))
                head_params.append(att_params)
            att_params = (head_params, context_out_params)
            return batch_output_shape, (rnn_params, rnn_params_dec, att_params, readout_params)
        elif rnn_specs['arch'] in ['enc_dec_att',]: # Old learnable attention matrix
            _, att_params = attention_init(k3, (phrase_length,))
            return batch_output_shape, (rnn_params, rnn_params_dec, att_params, readout_params)

    def apply_fun_enc_dec(params, inputs, in_masks, step=0, targets=None, returns={}):
        """Applies the RNN using encoder-decoder structure with different cells for both encoder and decoder and with attention mechanism"""
        
        if rnn_specs['arch'] == 'enc_dec_short':
            rnn_params_enc, rnn_params_dec, readout_params = params
            # Builds attend matrix based on shorts
            att_params = jnp.diag(jnp.asarray([int(i in rnn_specs['shorts']) for i in range(phrase_length)]))
        elif rnn_specs['arch'] in ('enc_dec_att', 'enc_dec_attl'): # Architectures with learnable attention
            rnn_params_enc, rnn_params_dec, att_params, readout_params = params
            attn_apply = functools.partial(attention_apply, att_params)
        elif rnn_specs['arch'] in ('enc_dec_attmh',): # Architectures with learnable attention
            rnn_params_enc, rnn_params_dec, att_params, readout_params = params
            head_params, context_out_params = att_params
            attn_apply = []
            for idx, apply_fn in enumerate(attention_applys):
                attn_apply.append(functools.partial(apply_fn, head_params[idx]))
            # Appends the context out apply to attn_apply too
            attn_apply.append(functools.partial(context_out_apply, context_out_params))
        elif rnn_specs['arch'] == 'enc_dec_atth': # Architecture with fixed attention
            rnn_params_enc, rnn_params_dec, readout_params = params
            attn_apply = attention_apply

        # Only gets initial state from encoder, [batch, len, dim]
        initial_states = cell.get_initial_state(rnn_params_enc, batch_size=inputs.shape[0])

        batch_apply_enc = functools.partial(cell.batch_apply, rnn_params_enc)
        batch_apply_dec = functools.partial(cell_dec.batch_apply, rnn_params_dec)

        rnn_apply = batch_apply_enc, batch_apply_dec

        if rnn_specs['arch'] in ['enc_dec_short', 'enc_dec_att']:
            # Softmax across dimension multiplying encoder sequence
            att_apply = functools.partial(attention_apply, jax.nn.softmax(att_params, axis=0))
            outs = encode_decode_attn(initial_states, inputs, rnn_apply, att_apply,
                                      functools.partial(readout_apply, readout_params), zero_context=rnn_specs['zero_context'])
        elif rnn_specs['arch'] in ('enc_dec_atth', 'enc_dec_attl', 'enc_dec_attmh'):
            dec_seq_size = (inputs.shape[0], rnn_specs['out_len'], rnn_specs['output_size']) # (batch size, phrase len, output word size)
            outs = encode_decode_attnh(initial_states, inputs, in_masks, rnn_apply, attn_apply, 
                                       functools.partial(readout_apply, readout_params), rnn_specs=rnn_specs, step=step,
                                       dec_seq_size=dec_seq_size, targets=targets, returns=returns)
        return outs

    def mask_fun(sequences, last_index):
        """Selects the last valid timestep from a batch of padded sequences."""
        # last_index.shape = (batch_size, phrase_length)
        last_index = last_index[:, :, jnp.newaxis]
        # sequences.shape = (batch_size, phrase_length, vocab_size)
        return jnp.take_along_axis(sequences, last_index, axis=1)

    def mask_sequences(sequences, last_index):
        """Set positions beyond the length of each sequence to 0. For sequences of shape (batch_size, seq_len, vocab_size)"""
        # sequences.shape = (batch_size, seq_length, vocab_size)
        # last_index.shape = (batch_size)
        mask = last_index[:, jnp.newaxis] >= jnp.arange(sequences.shape[1])
        return sequences * jnp.repeat(mask[:,:,jnp.newaxis], sequences.shape[-1], axis=2)

    def mask_sequences_one(sequences, last_index):
        """Set positions beyond the length of each sequence to 0. For sequences of shape (batch_size, seq_len)"""
        # sequences.shape = (batch_size, seq_length)
        # last_index.shape = (batch_size)
        mask = last_index[:, jnp.newaxis] >= jnp.arange(sequences.shape[1])
        return sequences * mask

    def sigmoid_xent_with_logits(logits, labels):
        return jnp.maximum(logits, 0) - logits * labels + \
                jnp.log(1 + jnp.exp(-jnp.abs(logits)))

    def xe_loss_fn(logits, labels, lengths):
        """
        Contains both log softmax and negative log liklihood loss
        """
        # Shifts maximum to zero
        shifted = logits - jax.lax.stop_gradient(logits.max(axis=-1, keepdims=True))
        # Log(e^x_i/sum_I e^x_I) = x_i - Log(sum_I e^x_I)
        log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

        labels_max = jnp.argmax(labels, axis=2)
        
        logliklihood = jnp.take_along_axis(log_probs, jnp.expand_dims(labels_max, axis=2), axis=2)
        
        # Masks the cross entropy loss
        xe_loss = -1 * np.mean(mask_sequences_one(jnp.squeeze(logliklihood), lengths))

        return xe_loss

    def loss_fun(params, batch, step=0):
        """Cross-entropy loss function."""
        all_logits = apply_fun(params, batch['inputs'], batch['in_index'], step=step, targets=batch['labels'])

        flatten = lambda params: jax.flatten_util.ravel_pytree(params)[0]
        l2_loss = rnn_specs['l2reg'] * np.sum(flatten(params)**2)
        
        # xe_loss = xe_loss_fn(logits, labels)
        xe_loss = xe_loss_fn(all_logits, batch['labels'], batch['out_index'])

        loss = l2_loss + xe_loss
        # Average over the batch
        return loss

    @jax.jit
    def accuracy_fun(params, batch):
        all_logits = apply_fun(params, batch['inputs'], batch['in_index'], targets=batch['labels'])

        logits = mask_sequences(all_logits, batch['out_index'])
        labels = mask_sequences(batch['labels'], batch['out_index'])
        
        predictions = jnp.argmax(logits, axis=2).astype(jnp.int32)
        labels_max = jnp.argmax(labels, axis=2).astype(jnp.int32)
        # Returns both accs for entire phrase and individual words
        return np.all(labels_max == predictions, axis=1), labels_max == predictions

    # Build the RNN cells.
    cell = build_cell_rnn(num_units)
    cell_dec = build_cell_rnn(num_units)
    readout_init, readout_apply = build_readout(num_outputs)
    apply_fun = apply_fun_enc_dec
    
    if rnn_specs['arch'] in ('enc_dec_atth'): # Dot product attention (not learnable)
        attention_init, attention_apply = build_dot_attention(rnn_specs)
    elif rnn_specs['arch'] in ('enc_dec_attl'): # Query, key, value attention (learnable)
        attention_init, attention_apply = build_qkv_attention(rnn_specs)
    elif rnn_specs['arch'] in ('enc_dec_attmh'): # Multiheaded attention (uses qkv)
        attention_inits = []
        attention_applys = []
        for _ in range(rnn_specs['attn_heads']):
            attention_init, attention_apply = build_qkv_attention(rnn_specs)
            attention_inits.append(attention_init)
            attention_applys.append(attention_apply)
        # Context_out is the linear layer that acts on the multiple heads' concatenated values
        context_out_init, context_out_apply = build_readout(rnn_specs['hidden_size'])
    elif rnn_specs['arch'] in ['enc_dec_short', 'enc_dec_att']: # Uses attention with fixed shorts or learnable parameters (OLD)
        attention_init, attention_apply = build_attention_old()
    else:
        raise ValueError('Rnn arch not recognized.')
        
    return init_fun, apply_fun, mask_sequences, loss_fun, accuracy_fun

def build_optimizer_step(optimizer, initial_params, loss_fun, gradient_clip=None):
    """Builds training step function."""

    # Destructure the optimizer triple.
    init_opt, update_opt, get_params = optimizer
    opt_state = init_opt(initial_params)

    @jax.jit
    def optimizer_step_noclip(current_step, state, batch):
        """Takes a single optimization step."""
        p = get_params(state)
        loss, gradients = jax.value_and_grad(loss_fun, argnums=0)(p, batch, current_step)
        new_state = update_opt(current_step, gradients, state)
        return current_step + 1, new_state, loss

    @jax.jit
    def optimizer_step_clip(current_step, state, batch):
        """Takes a single optimization step with clipped gradients."""
        p = get_params(state)
        loss, gradients = jax.value_and_grad(loss_fun, argnums=0)(p, batch, current_step)
        
        gradients = optimizers.clip_grads(gradients, gradient_clip)

        new_state = update_opt(current_step, gradients, state)
        return current_step + 1, new_state, loss    
    
    if gradient_clip is None:
        return opt_state, optimizer_step_noclip
    else:
        return opt_state, optimizer_step_clip

def Dense_nobias(out_dim, W_init=jax.nn.initializers.glorot_normal()):
    """Layer constructor function for a dense (fully-connected) layer without bias."""
    def init_fun(rng, input_shape):
        # input shape = (seq_length, hidden_dims)
        # output shape = (seq_length, out_dim)
        output_shape = input_shape[:-1] + (out_dim,)
        k1, _ = jax.random.split(rng)
        W = W_init(k1, (input_shape[-1], out_dim))
        return output_shape, (W)
    def apply_fun(params, inputs, **kwargs):
        W = params
        return jnp.dot(inputs, W)
    return init_fun, apply_fun

def dot_attention(output_dim):
    """ Constructor function for dot product attention (note this has no parameters itself) 
                output_dim: shape of values to be output (the same as the hidden dimension)
    """
    def init_fun(input_shape):
        # input_shape: shape of inputs to be fed into query, key, and value matrices
        output_shape = (output_dim,)
        return output_shape

    def apply_fun(dec_state, enc_seq, mask_fun, **kwargs):
        """
            dec_state.shape: (batch, hidden)
            enc_seq.shape: (batch, seq_len, hidden)
        """
        #  (batch, hidden_dim) x (batch, seq_len, hidden_dim) -> (batch, seq_len)
        aligns = jnp.einsum('ij,ikj->ik', dec_state, enc_seq)
        # aligns = jnp.einsum('ij,ikj->ik', dec_state, enc_seq) / np.sqrt(output_dim)
        attention_row = jax.nn.softmax(mask_fun(aligns), axis=1)

        # (batch, seq_len) x (batch, seq_len, value_dim) -> (batch, value_dim)
        context_vector = jnp.einsum('ij,ijk->ik', attention_row, enc_seq)

        return context_vector, jnp.swapaxes(attention_row, 0, 1)
    return init_fun, apply_fun

def qkv_attention(value_dim, qk_dim, W_init=jax.nn.initializers.glorot_normal()):
    """ Constructor function for a learnable attention matrix via queries, keys, and values 
                value_dim: shape of values to be output
                qk_dim: query and key dime
    """
    def init_fun(rng, input_shape):
        # input_shape: shape of inputs to be fed into query, key, and value matrices
        output_shape = (value_dim,)
        k0, k1, k2 = jax.random.split(rng, 3)
        q_mat = W_init(k0, (input_shape[0], qk_dim))
        k_mat = W_init(k1, (input_shape[0], qk_dim))
        v_mat = W_init(k2, (input_shape[0], value_dim))
        return output_shape, (q_mat, k_mat, v_mat)

    def apply_fun(params, dec_state, enc_seq, mask_fun, **kwargs):
        """
            dec_state.shape: (batch, hidden)
            enc_seq.shape: (batch, seq_len, hidden)
        """

        q_mat, k_mat, v_mat = params
        query = jnp.matmul(dec_state, q_mat)
        # Key calculation could be moved out of this since this is redundant
        keys = jnp.matmul(enc_seq, k_mat)
        values = jnp.matmul(enc_seq, v_mat)
        #  (batch, qk_dim) x (batch, seq_len, qk_dim) -> (batch, seq_len)
        aligns = jnp.einsum('ij,ikj->ik', query, keys) / np.sqrt(value_dim)
        # aligns = jnp.einsum('ij,ikj->ik', query, keys) 

        attention_row = jax.nn.softmax(mask_fun(aligns), axis=1)

        # (batch, seq_len) x (batch, seq_len, value_dim) -> (batch, value_dim)
        context_vector = jnp.einsum('ij,ijk->ik', attention_row, values)

        return context_vector, jnp.swapaxes(attention_row, 0, 1)
    return init_fun, apply_fun

def attention_old(out_dim, W_init=jax.nn.initializers.glorot_normal()):
    """ Constructor function for old version of learnable attention matrix (this was just learnable shorts) """
    def init_fun(rng, input_shape):
        # input shape = (# encoders,)
        # output shape = (# decoders,)
        output_shape = (out_dim,)
        k1, _ = jax.random.split(rng)
        att_mat = W_init(k1, (input_shape[0], out_dim))
        return output_shape, (att_mat)
    def apply_fun(params, inputs, **kwargs):
        att_mat = params
        return jnp.dot(inputs, att_mat)
    return init_fun, apply_fun

################################################################################
################################################################################
######################## Synthetic/scan data functions #########################
################################################################################
################################################################################

def generate_data(dataset_size, toy_params, rnn_specs):
    """
    Generate training data in numpy and then converts to JAX arrays
    """

    out_size = rnn_specs['output_size']

    syn_out_phrases = []
    syn_in_phrases = []
    syn_targets_np = np.zeros((dataset_size, toy_params['phrase_length'], out_size))
    syn_inputs_np = np.zeros((dataset_size, toy_params['phrase_length'], len(toy_params['in_words'])))
    # syn_target_masks_np = []
    syn_target_masks_np = np.zeros((dataset_size,))
    syn_input_masks_np = np.zeros((dataset_size,))

    start_time = time.time()
    for trial in range(dataset_size):
        out_phrase, in_phrase, out_phrase_tensor, in_phrase_tensor, out_target_mask, in_target_mask = toy.randomTrainingExample(toy_params)
        
        syn_targets_np[trial, :, :] = out_phrase_tensor
        syn_inputs_np[trial, :, :] = in_phrase_tensor
        syn_target_masks_np[trial] = out_target_mask
        syn_input_masks_np[trial] = in_target_mask
        # syn_target_masks_np.append(target_mask)

    print('Sythentic data generated in: {:0.2f} sec.'.format(time.time() - start_time))

    # Converts to JAX arrays 
    syn_target_masks = jnp.asarray(syn_target_masks_np, dtype=jnp.int32)
    # syn_target_masks =  syn_target_masks_np

    syn_data = {
        'inputs': jnp.asarray(syn_inputs_np)  ,  # Phrase tensors: dataset_size x phrase_len x in_dim
        'labels': jnp.asarray(syn_targets_np), # Sentiment tensors: dataset_size x phrase_len x out_dim
        'in_index': jnp.asarray(syn_input_masks_np, dtype=jnp.int32), # Target mask: list of integers up to phrase_len
        'out_index': jnp.asarray(syn_target_masks_np, dtype=jnp.int32), # Target mask: list of integers up to phrase_len
    }

    return syn_data

def shuffle_data(syn_data):
    """ Shuffles synthetic data for different epochs """
    dataset_size = syn_data['inputs'].shape[0]
    shuf_idxs = np.asarray(range(dataset_size), dtype=jnp.int32)
    np.random.shuffle(shuf_idxs)

    syn_data['inputs'] = syn_data['inputs'][shuf_idxs]
    syn_data['labels'] = syn_data['labels'][shuf_idxs]
    syn_data['in_index'] = syn_data['in_index'][shuf_idxs]
    syn_data['out_index'] = syn_data['out_index'][shuf_idxs]

    return syn_data

def override_data(path, save_file):
    if save_file and os.path.exists(path):
        print('File already exists at:', path)
        override = input('Override? (Y/N):')
        if override == 'Y':
            save_file = True
        else:
            save_file = False

    return save_file

def save_run_data(path, save_file, params_jax, toy_params, rnn_specs, train_params, save_loss=False, loss_data=[]):

    if save_file:
        with open(path, 'wb') as save_file:
            pickle.dump(params_jax, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(toy_params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(rnn_specs, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_params, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data Saved')
        # if save_loss:
        #   dot_idx = path.index('.')
        #   new_path = path[:dot_idx] + '_loss' + path[dot_idx:]
        #   with open(new_path, 'wb') as save_file:
        #     pickle.dump(loss_data, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        #   print('Loss Data Saved')
    else:
        print('Data Not Saved')

def test_accuracy(params, test_batch, accuracy_fun):
    _phrase_accs, _accs = accuracy_fun(params, test_batch)

    return np.mean(_phrase_accs), np.mean(_accs)

def train_on_synthetic_data(opt_state, step_fun, get_params, accuracy_fun, train_params, toy_params, rnn_specs):
    """ Generate sythetic and test data and train for some number of epochs"""
    
    train_set_size = train_params['train_set_size']
    test_set_size = train_params['test_set_size']
    num_batchs = train_params['num_batchs']
    global_step = train_params['global_step']
    total_steps = train_params['total_steps']
    train_losses = train_params['train_losses']
    decay_fun = train_params['decay_fun']
    print_every = train_params['print_every']
    batch_size = train_params['batch_size']

    for dataset in range(train_params['n_datasets']):
        # Generates new train/test datasets 
        syn_data = generate_data(train_set_size, toy_params, rnn_specs)
        syn_data_test = generate_data(test_set_size, toy_params, rnn_specs)

        # Sets output sequence length to length of output sequence
        rnn_specs['out_len'] = syn_data['labels'].shape[1]

        for epoch in range(train_params['epochs']):
            # Shuffles training data after first epoch
            if epoch > 0:
                print('Shuffling Data')
                syn_data = shuffle_data(syn_data)

            print('Running batches {} to {}'.format(global_step, global_step+num_batchs))
            start_time = time.time()  
            avg_loss = 0 
            for step in range(total_steps, total_steps+num_batchs):
                batch = step - total_steps 
                batch_data = {
                    'inputs': syn_data['inputs'][batch*batch_size:(batch+1)*batch_size, :, :], # In phrase tensors: batch x phrase_len x in_dim
                    'labels': syn_data['labels'][batch*batch_size:(batch+1)*batch_size, :, :], # Out phrase tensors: batch x phrase_len x out_dim
                    'in_index': syn_data['in_index'][batch*batch_size:(batch+1)*batch_size], # In phrase mask: list of integers up to phrase_len
                    'out_index': syn_data['out_index'][batch*batch_size:(batch+1)*batch_size], # Out phrase mask: list of integers up to phrase_len
                }

                global_step, opt_state, loss = step_fun(global_step, opt_state, batch_data)
                train_losses.append(loss)
                avg_loss += loss

                if (global_step+1) % print_every == 0:
                    phrase_acc, word_acc = test_accuracy(get_params(opt_state), syn_data_test, accuracy_fun)

                    batch_time = time.time() - start_time
                    step_size = decay_fun(global_step)
                    s = "Step {}, step size: {:0.5f}, phrase acc {:0.4f}, word acc {:0.4f}, avg training loss {:0.4f}"
                    print(s.format(global_step+1, step_size, phrase_acc, word_acc, avg_loss/print_every))
                    start_time = time.time()
                    avg_loss = 0
                # if (global_step+1) % 500 == 0:
                #   print('Positional Encoding:', rnn_specs['pos_enc_params']['pos_enc'])
                # if train_params['init_pos_enc'] and rnn_specs['pos_enc_params']['pos_enc'] != None: # Optional removal of positional encoding at a certain global step
                #   if global_step > train_params['remove_pos_enc_batch']:
                #     print('Removing positional encoding')
                #     rnn_specs['pos_enc_params']['pos_enc'] = None

            train_params['total_steps'] += num_batchs
    
    train_params['global_step'] = global_step
    train_params['train_losses'] = train_losses

    return opt_state, train_params

def train_on_scan_data(scan_data, opt_state, step_fun, get_params, accuracy_fun, train_params, toy_params, rnn_specs, raw_phrases):
    """ Train SCAN data (imported elsewhere) for some number of epochs"""
    
    train_set_size = train_params['train_set_size']
    test_set_size = train_params['test_set_size']
    global_step = train_params['global_step']
    total_steps = train_params['total_steps']
    train_losses = train_params['train_losses']
    decay_fun = train_params['decay_fun']
    print_every = train_params['print_every']
    batch_size = train_params['batch_size']

    raw_in_phrases, raw_out_phrases = raw_phrases

    scan_params = toy_params['scan_params']
    
    for dataset in range(train_params['n_datasets']):
        # Generate scan data (normal or extended, for latter doesn't use trial count)
        scan_data, scan_params = scan.generate_scan(raw_in_phrases, raw_out_phrases, scan_params, trials=train_params['train_set_size'])
        
        # max_in = scan_data['inputs'].shape[1]
        # max_out = scan_data['labels'].shape[1]
        # print(f'Max in: {max_in} // Max out: {max_out} (post-pad)')

        # print('Scan input shape:', scan_data['inputs'].shape)
        # print('Scan labels shape:', scan_data['labels'].shape)
        # print('Scan indexs shape:', scan_data['index'].shape)
        
        if scan_params['extend_scan']:  # Generates new test set for extended scan
            scan_test_data, scan_params = scan.generate_scan(raw_in_phrases, raw_out_phrases, scan_params, trials=train_params['test_set_size'])
            num_batchs = train_params['num_batchs']
        else:
            # Handles incomplete batches
            num_batchs = int(np.floor(scan_data['inputs'].shape[0]/batch_size))
            print(f'Running {num_batchs} batches per epoch.')
            # For now, just uses full SCAN set as test set when not extending
            scan_test_data = scan_data
        # Sets output sequence length to length of output sequence
        rnn_specs['out_len'] = scan_test_data['labels'].shape[1]

        for epoch in range(train_params['epochs']):
            # Shuffles training data after first epoch
            if epoch > 0:
                scan_data = shuffle_data(scan_data)

            start_time = time.time()  
            avg_loss = 0 
            for step in range(total_steps, total_steps+num_batchs):
                batch = step - total_steps 
                batch_data = {
                    'inputs': scan_data['inputs'][batch*batch_size:(batch+1)*batch_size, :, :], # In phrase tensors: batch x phrase_len x in_dim
                    'labels': scan_data['labels'][batch*batch_size:(batch+1)*batch_size, :, :], # Out phrase tensors: batch x phrase_len x out_dim
                    'in_index': scan_data['in_index'][batch*batch_size:(batch+1)*batch_size], # Out phrase mask: list of integers up to phrase_len
                    'out_index': scan_data['out_index'][batch*batch_size:(batch+1)*batch_size], # Out phrase mask: list of integers up to phrase_len
                }

                global_step, opt_state, loss = step_fun(global_step, opt_state, batch_data)
                train_losses.append(loss)
                avg_loss += loss

                if (global_step+1) % print_every == 0:
                    phrase_acc, word_acc = test_accuracy(get_params(opt_state), scan_test_data, accuracy_fun)

                    batch_time = time.time() - start_time
                    step_size = decay_fun(global_step)
                    s = "Step {}, step size: {:0.5f}, phrase acc {:0.4f}, word acc {:0.4f}, avg training loss {:0.4f}"
                    print(s.format(global_step+1, step_size, phrase_acc, word_acc, avg_loss/print_every))
                    start_time = time.time()
                    avg_loss = 0

            train_params['total_steps'] += num_batchs
    
    train_params['global_step'] = global_step
    train_params['train_losses'] = train_losses

    return opt_state, train_params

def default_params(train_params, toy_params, rnn_specs):
    """ Sets the defaults of many parameters if not specified"""

    if 'rules' not in toy_params: toy_params['rules'] = []
    if 'loss_type' not in toy_params: toy_params['loss_type'] = 'XE'

    if 'shorts' not in rnn_specs: rnn_specs['shorts'] = ()
    if 'cell_type' not in rnn_specs: rnn_specs['cell_type'] = 'GRU'
    if 'zero_rec' not in rnn_specs: 
        if 'zero_context' in rnn_specs:
            rnn_specs['zero_rec'] = rnn_specs['zero_context']
        else:
            rnn_specs['zero_rec'] = False
    if 'zero_decoder_inputs' not in rnn_specs: rnn_specs['zero_decoder_inputs'] = False
    if 'init_pos_enc' not in rnn_specs: rnn_specs['init_pos_enc'] = False
    if 'remove_pos_enc_step' not in rnn_specs: rnn_specs['remove_pos_enc_step'] = 0
    if 'l2reg' not in rnn_specs: rnn_specs['l2reg'] = train_params['l2reg']
    if 'var_length' not in rnn_specs: rnn_specs['var_length'] = toy_params['var_length']
    if 'teacher_force' not in rnn_specs: rnn_specs['teacher_force'] = False

    if 'qk_size' not in rnn_specs: rnn_specs['qk_size'] = rnn_specs['hidden_size']
    if 'v_size' not in rnn_specs: rnn_specs['v_size'] = rnn_specs['hidden_size']

    if 'attn_heads' not in rnn_specs: rnn_specs['attn_heads'] = 2

    if rnn_specs['var_length']: rnn_specs['in_eos_idx'] = toy_params['in_words'].index('<pad>')

    if 'pos_enc_params' not in rnn_specs: rnn_specs['pos_enc_params'] = {}

    {'pos_enc': True, 'time_scale': 50.0, 'amplitude': 1.0}

    if 'pos_enc' not in rnn_specs['pos_enc_params']: rnn_specs['pos_enc_params']['pos_enc'] = False
    if 'time_scale' not in rnn_specs['pos_enc_params']: rnn_specs['pos_enc_params']['time_scale'] = 100.0
    if 'amplitude' not in rnn_specs['pos_enc_params']: rnn_specs['pos_enc_params']['amplitude'] = 1.0
    if 'rand_seq_offset' not in rnn_specs['pos_enc_params']: rnn_specs['pos_enc_params']['rand_seq_offset'] = False

    return train_params, toy_params, rnn_specs


################################################################################
################################################################################
################## Functions for encoder->decoder structure ####################
################################################################################
################################################################################


def identity(x):
    """Identity function f(x) = x."""
    return x

def encode_decode_attn(initial_states, input_sequences, rnn_update, attention_apply, readout=identity,
                       return_hidden=False, zero_context=False):
    """ 
    Encoder decoder structure which implements attention using a learnable attention layer. This 
    attention layer passes a context to the INPUT of the RNN by concatenating it with the hidden state.
    """

    hidden_state_keep = int(1/2*initial_states.shape[-1])
    # print('Keeping first {} of {} hidden'.format(hidden_state_keep, initial_states.shape[-1]))

    def _step_e(state, inputs):
        """ Encoder step, output is simply hidden states 
        state = (batch_size, 2*hidden_size)
        inputs = (batch_size, input_size)
        """
        if zero_context:
            state = jnp.zeros(state.shape)
        # Zeros second half of the hidden state
        state = jnp.concatenate([state[:,:hidden_state_keep], jnp.zeros((state.shape[0], hidden_state_keep))], axis=1)

        next_state = enc_update(inputs, state)
        outputs = identity(next_state[:,:hidden_state_keep])
        return next_state, outputs

    def _step_d(carry, seq_input):
        """ Decoder step, output goes through readouts now """
        state, inputs = carry # unpack carry
        if zero_context:
            print('Zeroing Context')
            state = jnp.zeros(state.shape)
        # Concatanate seq_input with hidden to get new hidden (for attention)
        state = jnp.concatenate([state[:,:hidden_state_keep], seq_input], axis=1)

        next_state = dec_update(inputs, state)
        # Only first half of hidden state goes to readout
        outputs = readout(next_state[:,:hidden_state_keep])
        
        # Converts outputs into one-hot vectors to pass back into network
        output_tensors = jax.nn.one_hot(jnp.argmax(outputs, axis=1), outputs.shape[1])
        carry = next_state, output_tensors
        
        if not return_hidden:
            return carry, outputs
        else:
            return carry, next_state[:,:hidden_state_keep]

    enc_update, dec_update = rnn_update

    # Run encoder
    input_sequences = jnp.swapaxes(input_sequences, 0, 1)
    encode_final, enc_outs = jax.lax.scan(_step_e, initial_states, input_sequences)

    eos_input = jnp.zeros(input_sequences[0].shape)
    init_val = (encode_final, eos_input)

    # Run decoder
    input_seq_hidden = enc_outs # Hidden states of encoder (enc_seq_len, batch_size, hidden_size)

    # Multiplies hidden states by the attend matrix (with some swap axes to get everything the right shape)
    input_seq = jnp.swapaxes(attention_apply(jnp.swapaxes(input_seq_hidden, 0, 2)), 0, 2)

    _, dec_outs = jax.lax.scan(_step_d, init_val, input_seq)

    if not return_hidden:
        return jnp.swapaxes(dec_outs, 0, 1)
    else:
        full_hs = jnp.concatenate((enc_outs, dec_outs), axis=0)
        return jnp.swapaxes(full_hs, 0, 1)

def encode_decode_attnh(initial_states, input_sequences, input_masks, rnn_update, attn_apply, readout=identity,
                        rnn_specs={}, dec_seq_size=None, step=0, targets=None, returns={}):
    """ 
    Encoder decoder structure with attention, this time using the hidden state overlap to find attention.
    Here the attention context vector gets concatenated with the decoder RNN output and passed to the readout.
    
    input_sequences: shape (batch_size, in_seq_len, input_dim)
    input_masks: shape (batch_size)
    dec_seq_size: (batch_size, out_seq_len, output_dim)
    targets: shape (batch_size, out_seq_len, output_dim)
    
    """
    if rnn_specs['cell_type'] != 'LSTM': # Keeping all of the hidden state for non-LSTM
        hidden_state_keep = initial_states.shape[-1]
    else: # For LSTM, we only send the first half of the "hidden state" to the readout, because this corresponds to the true hidden state (i.e. not the cell part)
        hidden_state_keep = int(1/2*initial_states.shape[-1])

    if rnn_specs['var_length']: eos_idx = rnn_specs['in_eos_idx']
    # if train_params['init_pos_enc'] and rnn_specs['pos_enc_params']['pos_enc']: # Conditions for changing positional encoding
    #   rnn_specs['pos_enc_params']['pos_enc'] = jax.lax.cond(step > train_params['remove_pos_enc_step'], switch_pos_enc, 
    #                                                         lambda _: True, operand=None)

    def switch_pos_enc(_):
        """ Function for removing positional encoding """
        print('Removing positional encoding')
        return False
    def zero_pos_enc(input_vals):
        seq, _, amp, _ = input_vals

        # print('Removing positional encoding')
        # pos_enc = jnp.zeros(seq.shape[1:])
        # return jnp.repeat(pos_enc[jnp.newaxis, :, :], seq.shape[0], axis=0)

        pos_enc_vals = positional_encoding(input_vals)
        amp_mod = jnp.max(jnp.array((0.0, (2000 - (step - train_params['remove_pos_enc_step']))/2000)))
        return amp_mod * pos_enc_vals
    
    def mask_aligns(sequences, last_index=input_masks):
        """Set positions beyond the length of each sequence to large negative number. For sequences of shape (seq_len, batch_size)"""
        alpha = 10000
        mask = last_index[:, jnp.newaxis] >= jnp.arange(sequences.shape[1])
        mask_default = alpha * (mask - jnp.ones(mask.shape)) # alpha for anything that doesn't get past the mask
        return sequences * mask + mask_default

    def _step_e(carry, inputs):
        """ Encoder step, output is simply hidden states 
        state = (batch_size, hidden_size)
        seq_input = (batch_size, input_size)
        """
        old_state, is_eos = carry
        seq_input, pos_enc_vals = inputs

        if rnn_specs['zero_rec']: # Zeros out recurrent portion of input
            next_state = enc_update(seq_input + pos_enc_vals, jnp.zeros(old_state.shape))
        else:
            next_state = enc_update(seq_input + pos_enc_vals, old_state)
        # determines which state to carry to next input based on whether EoS has occurred
        carried_state = jnp.where(is_eos[:, np.newaxis], old_state, next_state)
        outputs = identity(carried_state)
        
        # Update is_eos (true if one-hot in eos_idx)
        if rnn_specs['var_length']: is_eos = jnp.logical_or(is_eos, np.argmax(seq_input, axis=1) == eos_idx)
        
        return (carried_state, is_eos), outputs

    def _step_d(carry, inputs):
        """ Decoder step, output goes through readouts now """
        old_state, seq_input = carry # unpack carry into hidden sate and previous input
        pos_enc_vals, targets = inputs # unpack inputs

        if rnn_specs['teacher_force']:
            dec_input = targets # Uses shifted targets as input
            print('Using targets as decoder input')
        else:
            dec_input = seq_input # Uses previous input 

        if rnn_specs['zero_rec']: # Zeros out recurrent portion of input (for AO)
            next_state = dec_update(dec_input + pos_enc_vals, jnp.zeros(old_state.shape))
        else:
            next_state = dec_update(dec_input + pos_enc_vals, old_state)

        if rnn_specs['zero_attention']: # Attention not used, only decoder output goes to readout
            outputs = readout(next_state[:, :hidden_state_keep])
        else: # Attention calculations
            dec_state = next_state[:, :hidden_state_keep]
            enc_seq = jnp.swapaxes(input_seq_hidden[:, :, :hidden_state_keep], 0, 1)  # Returns to (batch, seq_len, hidden)
            mask_fun = functools.partial(mask_aligns, last_index=input_masks)
            
            if rnn_specs['arch'] not in ('enc_dec_attmh'):  # Single head attentions
                context_vector, attention_row = attn_apply(dec_state, enc_seq, mask_fun)
            else: # Multiheaded attentions
                c_vecs = []
                attention_rows = [] # This will be (heads, batch, enc_len)
                for head_idx in range(rnn_specs['attn_heads']):
                    context_vec, attention_row = attn_apply[head_idx](dec_state, enc_seq, mask_fun)
                    c_vecs.append(context_vec)
                    attention_rows.append(attention_row)
                attention_row = jnp.transpose(jnp.array(attention_rows), axes=[1, 2, 0]) # (heads, enc_idx, batch) -> (enc_idx, batch, heads)

                context_total = jnp.concatenate(c_vecs, axis=1)
                context_vector = attn_apply[-1](context_total)  # Last apply corresponds to context_out linear layer

            if rnn_specs['zero_rec']: # Only attention state goes to readout
                outputs = readout(context_vector)
            else: # Concatenated hidden state and attention state go to readout
                outputs = readout(jnp.concatenate([next_state[:, :hidden_state_keep], context_vector], axis=1))
        
        if rnn_specs['zero_decoder_inputs']: # Just feeds zeros into decoder
            # next_seq_input = 1/outputs.shape[1] * jnp.ones(outputs.shape)
            next_seq_input = jnp.zeros(outputs.shape)
            # next_seq_input = jax.ops.index_update(next_seq_input, jax.ops.index[:, 0], jnp.ones(outputs.shape[0]))
            # key = jax.random.PRNGKey(23232)
            # key, subkey = jax.random.split(key)
            # rand_ints = jax.random.randint(subkey, outputs.shape[0], 0, outputs.shape[1])
            # next_seq_input = jax.nn.one_hot(rand_ints, outputs.shape[1])
        else: # Converts outputs into one-hot vectors to pass to next step
            next_seq_input = jax.nn.one_hot(jnp.argmax(outputs, axis=1), outputs.shape[1])
        carry = next_state, next_seq_input
        
        # Constructs outputs, which could be many things depending on desired return
        if returns == {}:
            return carry, outputs
        else:
            outs = [outputs,]
            if 'att_matrix' in returns: outs.append(attention_row)
            if 'hidden' in returns: outs.append(next_state)
            if 'context_vector' in returns: outs.append(context_vector)
            if 'context_total' in returns: outs.append(context_total)
            return carry, tuple(outs)

    pos_enc_params = rnn_specs['pos_enc_params']
    
    enc_update, dec_update = rnn_update
    is_eos = jnp.zeros(input_sequences.shape[0], dtype=np.bool) # (batch_size)
    initial_carry = (initial_states, is_eos)
    
    # Modification of targets for use in teacher forcing
    if targets is None or not rnn_specs['teacher_force']: # Just sets targets to zero, since they won't be used
        targets = jnp.zeros(dec_seq_size)
    else: # Updates targets for use in teacher forcing
        targets = jnp.roll(targets, axis=1, shift=1) # shifts targets 1 forward along dec_seq axis
        # inserts <BoS> at seq_idx=0 so can be used as inputs for teacher forcing
        # targets = jax.ops.index_update(targets, jax.ops.index[:, 0, :], jnp.zeros((targets.shape[0], targets.shape[2]))) # Zeros
        targets = jax.ops.index_update(targets, jax.ops.index[:, 0, :], 1/dec_seq_size[2] * jnp.ones((dec_seq_size[0], dec_seq_size[2]))) # Average input
    # Adds a random offset to the positional encoding
    if pos_enc_params['rand_seq_offset']:
        offsets = np.random.randint(0, 20, size=(input_sequences.shape[0], 1))
        # key = jax.random.PRNGKey(123231)
        # offsets = jax.random.randint(key, (input_sequences.shape[0], 1), 0, 20)
        offsets = jnp.array(offsets)
        print('Adding random Positoinal encoding offsets (between 0 and 20)')
    else:
        offsets = jnp.zeros((input_sequences.shape[0], 1))

    #### Run encoder ####

    if pos_enc_params['pos_enc']: # Generates positional encoding
        pos_enc_input = input_sequences, pos_enc_params['time_scale'], pos_enc_params['amplitude'], pos_enc_params['rot'], pos_enc_params['rand_seq_offset'], offsets
        if rnn_specs['init_pos_enc']: # Positional encoding only at start of training
            pos_enc_vals = jax.lax.cond(step > rnn_specs['remove_pos_enc_step'], zero_pos_enc, positional_encoding, operand=pos_enc_input) 
        else: # Normal positional encoding
            pos_enc_vals = positional_encoding(pos_enc_input)
    else:
        pos_enc_vals = jnp.zeros(input_sequences.shape)

    pos_enc_vals = jnp.swapaxes(pos_enc_vals, 0, 1)
    input_sequences = jnp.swapaxes(input_sequences, 0, 1)
    # Keeps inputs and potential positional encoding values separate so EoS character still identifiable
    encode_final, enc_outs = jax.lax.scan(_step_e, initial_carry, (input_sequences, pos_enc_vals))
    encode_final_hs, _ = encode_final

    if dec_seq_size == None: # Assumes same dimension of output sequence as input sequence if left blank
        dec_seq_size = input_sequences.shape[:-1] + (rnn_specs['output_size'],)
    # Special character to start decoder is just average input  
    sos_input = 1/dec_seq_size[2] * jnp.ones((dec_seq_size[0], dec_seq_size[2])) # shape: (batch_size, decoder input dim)
    initial_carry = (encode_final_hs, sos_input)

    input_seq_hidden = enc_outs # Hidden states of encoder (used for attention) (enc_seq_len, batch_size, hidden_size)
    

    #### Run Decoder ####

    if pos_enc_params['pos_enc']: # Generates positional encoding
        pos_enc_input = jnp.zeros(dec_seq_size), pos_enc_params['time_scale'], pos_enc_params['amplitude'], pos_enc_params['rot'], pos_enc_params['rand_seq_offset'], offsets
        if rnn_specs['init_pos_enc']: # Positional encoding only at start of training
            pos_enc_vals = jax.lax.cond(step > rnn_specs['remove_pos_enc_step'], zero_pos_enc, positional_encoding, operand=pos_enc_input) 
        else: # Normal positional encoding
            pos_enc_vals = positional_encoding(pos_enc_input) 
    else:
        pos_enc_vals = jnp.zeros(dec_seq_size) 

    pos_enc_vals = jnp.swapaxes(pos_enc_vals, 0, 1)
    targets = jnp.swapaxes(targets, 0, 1) # for teacher forcing
    # _ = _step_d(initial_carry, (pos_enc_vals[0], targets[0]))
    _, dec_outs = jax.lax.scan(_step_d, initial_carry, (pos_enc_vals, targets))

    # Handles different types of returns
    if returns == {}: # Only output is raw logits
        return jnp.swapaxes(dec_outs, 0, 1)
    else:
        # First output is always the raw logits
        outs = [jnp.swapaxes(dec_outs[0] , 0, 1),]
        out_idx = 1
        if 'att_matrix' in returns:
            outs.append(jnp.swapaxes(dec_outs[out_idx], 0, 2))
            out_idx += 1
        if 'hidden' in returns:
            outs.append(jnp.swapaxes(jnp.concatenate((enc_outs, dec_outs[out_idx]), axis=0), 0, 1))
            out_idx += 1
        if 'context_vector' in returns:
            outs.append(jnp.swapaxes(dec_outs[out_idx], 0, 1))
            out_idx += 1
        if 'context_total' in returns:
            outs.append(jnp.swapaxes(dec_outs[out_idx], 0, 1))
            out_idx += 1
        return tuple(outs)

def positional_encoding(input_vals):
    """ 
    Returns positional encoding values. 
    Assumes seq dimensions are (batch, seq_len, word_space) 
    """
    # Unpacks input
    seq, scale, amp, rot, rand_seq_offset, offsets = input_vals
    word_dim = seq.shape[2]

    if not rand_seq_offset: # Standard positional encoding
        pos_enc = jnp.zeros(seq.shape[1:])

        for word_idx in range(word_dim):
            if word_idx % 2 == 0:
                # pos_enc[:, word_idx] = [amp*jnp.sin(seq_idx/(scale**(word_idx/word_dim))) for seq_idx in range(seq.shape[1])]
                pos_enc = jax.ops.index_update(pos_enc, jax.ops.index[:, word_idx], [amp*jnp.sin(seq_idx/(scale**(word_idx/word_dim))) for seq_idx in range(seq.shape[1])])
            else:
                # pos_enc[:, word_idx] = [amp*jnp.cos(seq_idx/(scale**((word_idx - 1)/word_dim))) for seq_idx in range(seq.shape[1])]
                pos_enc = jax.ops.index_update(pos_enc, jax.ops.index[:, word_idx], [amp*jnp.cos(seq_idx/(scale**((word_idx - 1)/word_dim))) for seq_idx in range(seq.shape[1])])

        if rot is None:
            rot = jnp.identity(pos_enc.shape[1])
        # Rotates positional encoding vectors (this is useful when using one-hot)
        pos_enc = jnp.matmul(pos_enc, rot)

        return jnp.repeat(pos_enc[jnp.newaxis, :, :], seq.shape[0], axis=0) 
    else: # Positoinal encoding with random sequence offsets for each batch (this does the positional encoding calculation much more efficiently)
        
        pos_enc = jnp.zeros(seq.shape)

        # Only does even word_dimensions
        scale_denoms = 1 / (scale**(jnp.arange(0, word_dim, 2)/word_dim))
        scale_numerators = offsets + jnp.arange(seq.shape[1])[jnp.newaxis, :]

        # scale_numerators
        sinusoid_args = jnp.einsum('ij, k -> ijk', scale_numerators, scale_denoms)

        pos_enc = jax.ops.index_update(pos_enc, jax.ops.index[:, :, ::2], amp*jnp.sin(sinusoid_args))
        pos_enc = jax.ops.index_update(pos_enc, jax.ops.index[:, :, 1::2], amp*jnp.cos(sinusoid_args))

        if rot is None:
            rot = jnp.identity(pos_enc.shape[1])
        # Rotates positional encoding vectors (this is useful when using one-hot)
        pos_enc = jnp.matmul(pos_enc, rot)

        return pos_enc


def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
            x = random_state.normal(size=(dim-n+1,))
            D[n-1] = np.sign(x[0])
            x[0] -= D[n-1]*np.sqrt((x*x).sum())
            # Householder transformation
            Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
            mat = np.eye(dim)
            mat[n-1:, n-1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H