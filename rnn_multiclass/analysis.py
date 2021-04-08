import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from synthetic import synthetic_data as syn

import sys
sys.path.append('/content/drive/My Drive/ml_research/fixedpoints_nlp/reverse-engineering-neural-networks')
import renn

def build_cell(rnn_specs):
  if 'cell_type' not in rnn_specs: rnn_specs['cell_type']  = 'GRU' # Default
  
  print('Using {} cell'.format(rnn_specs['cell_type']))
  if rnn_specs['cell_type'] == 'Linear':
    cell = renn.rnn.cells.VanillaRNNIdentity(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'VanillaReLU':
    cell = renn.rnn.cells.VanillaRNNReLU(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'Vanilla':
    cell = renn.rnn.cells.VanillaRNN(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'VanillaRes':
    cell = renn.rnn.cells.VanillaRNNRes(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'VanillaRes2':
    cell = renn.rnn.cells.VanillaRNNResTwo(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'VanillaSplit':
    split_idx = rnn_specs['split_idx'] if 'split_idx' in rnn_specs else np.floor(rnn_specs['hidden_size']/2)
    cell = renn.rnn.cells.VanillaRNNSplit(rnn_specs['hidden_size'], split_idx=split_idx)
  elif rnn_specs['cell_type'] == 'VanillaSplitReLU':
    split_idx = rnn_specs['split_idx'] if 'split_idx' in rnn_specs else np.floor(rnn_specs['hidden_size']/2)
    cell = renn.rnn.cells.VanillaRNNSplitReLU(rnn_specs['hidden_size'], split_idx=split_idx)
  elif rnn_specs['cell_type'] == 'GRU':
    cell = renn.GRU(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'LSTM':
    cell = renn.LSTM(rnn_specs['hidden_size'])
  elif rnn_specs['cell_type'] == 'UGRNN':
    cell = renn.UGRNN(rnn_specs['hidden_size'])
  return cell

# Some helper functions to collect RNN hidden states.

def _get_all_states_wrapper(inputs, cell, rnn_params):
  @jax.jit
  def _get_all_states(inputs, rnn_params):
    """Get RNN states in response to a batch of inputs (also returns masked states)."""
    initial_states = cell.get_initial_state(rnn_params, batch_size=inputs.shape[0])
    return renn.unroll_rnn(initial_states, inputs, functools.partial(cell.batch_apply, rnn_params))

  return _get_all_states(inputs, rnn_params)

def rnn_states(cell, batch, rnn_params, only_final=True):
  """Return (masked) RNN states."""
  states =  _get_all_states_wrapper(batch['inputs'], cell, rnn_params)
  if only_final: # returns only the final hidden state
    return [h[idx[0]] for h, idx in zip(states, batch['index'])]
  else:
    # return [h[:batch['index'][0]+1] for h in states]
    return [h[:idx[0]] for h, idx in zip(states, batch['index'])]

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

def find_m_k(k, dist_sq):
  """
  Calculates m_k from Levina, Bickel (Eq. (9))
  
  k: Number of nearest neighbors
  dist_sq: n x n matrix of Euclidean distances between points
  """

  n = dist_sq.shape[0]
  
  m_ks = np.zeros((n,))
  k_eff = k + 1 # Self-distance included, so bump k up by one

  for idx in range(n): # Calculate m_k(x) for all x
    idx_dists = dist_sq[idx, :]
    kidxs = np.argpartition(idx_dists, k_eff)
    max_kdist = np.max(idx_dists[kidxs[:k_eff]])
    sum_of_logs = 0
    for i in range(k_eff):
      if idx_dists[kidxs[i]] > 0: # Ignores self distance
        sum_of_logs += np.log(max_kdist / idx_dists[kidxs[i]])
    m_ks[idx] = 1 / (1/(k - 1) * sum_of_logs)

  return np.mean(m_ks)

import scipy
from scipy.spatial.distance import squareform

def find_correlation_dimension(r_val_range, dist_sq, num_pts=100, show_plot=False):
  """ 
  Calculates the linear regression fit of correlation dimension(r). The slope of said result is the estimated dimension

  r_val_range: log10(Range) of r values, specifically np.logspace(r[0], r[1]), 100)
  dist_sq: n x n matrix of Euclidean distances between points
  num_pts: number of points to fit

  Output: slope, intercept, r_value, p_value, std_err of linear regression fit
  """
  n = dist_sq.shape[0]
  distances = squareform(dist_sq)  # Changes to n * (n-1) / 2 vector
  r_values = np.logspace(r_val_range[0], r_val_range[1], num_pts)

  thresh_count = np.asarray([np.count_nonzero(distances < r_val) for r_val in r_values])
  c_n = 2/(n*(n - 1)) * thresh_count

  slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(r_values), np.log10(c_n))

  if show_plot: # Shows plot of fit
    plt.figure()
    plt.loglog(r_values, c_n, 'b')

    y_fit_values = np.asarray([10**intercept * r_val**slope for r_val in r_values])
    plt.loglog(r_values, y_fit_values, 'r')

    plt.xlabel('r_value')
    plt.ylabel('C_n')

  return slope, intercept, r_value, p_value, std_err

def local_pca_analysis(states, dist_sq, n_pts=100, k=100):
  """ Randomly chooses a bunhc of points and calculates local PCAs

  n_pts = number of points to randomly choose
  k = number of nearest neighbors to include

  """
  pr_vals = np.zeros((n_pts,))

  for trial in range(n_pts):
    pt_idx = np.random.randint(states.shape[0])
    nn_idxs = np.argpartition(dist_sq[pt_idx,:], k)[:k]
    pr_vals[trial] = local_pca(states[nn_idxs])

  return np.mean(pr_vals)

def local_pca(pca_states):
  """ Performs a local PCA on a subset of states, returns the participation ratio"""

  n_comp = np.min([pca_states.shape[0], pca_states.shape[1]])
  pca_local = PCA(n_components=n_comp)
  _ = pca_local.fit_transform(pca_states)
 
  return participation_ratio_vector(pca_local.explained_variance_)