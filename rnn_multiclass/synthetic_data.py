import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax

import numpy as np

import time


def generateWordBank(toy_params):
  """ Creates the word bank based on various inputs """

  context_words = toy_params['context_words']
  variable_length = toy_params['variable_length'] if 'variable_length' in toy_params else False
  unordered = toy_params['unordered_class']
  if unordered:
    words = []
    n_unordered = toy_params['n_unordered']
    for i in range(n_unordered):
      # words.extend(['kinda'+str(i), 'very'+str(i)])
      words.extend(['kinda'+str(i)])
      # words.extend(['kinda'+str(i), 'very'+str(i), 'bad'+str(i), 'awful'+str(i)])
  else:
    words = ['awful', 'bad', 'good', 'awesome']
  
  # words.extend(['the', 'the2', 'the3', 'the4', 'the5'])
  # words.extend(['the', 'the2', 'the3'])
  words.extend(['the'])

  if context_words:
    words.extend(['not', 'extremely'])

  return words

def make_toy_phrase(toy_params):
  """ Creates a single toy phrase from the word bank """
  
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']
  variable_length = toy_params['variable_length']
  min_phrase_len = toy_params['min_phrase_len']
  the_padding = toy_params['the_padding']

  phrase = []
  
  # Sets phrase length, allowing for variable length
  if variable_length:
    length = min_phrase_len + np.random.randint(phrase_length - min_phrase_len)
  else:
    length = phrase_length
  
  if the_padding:
    length -= 20
    for _ in range(10):
      phrase.append('<noise>')
  
  n_words = len(words)

  for idx in range(length):
    next_word = False
    while not next_word:
      next_word = False

      word_idx = np.random.randint(n_words)
      if words[word_idx] == 'extremely' and idx > 0: # No repeat 'extremely'
        if phrase[idx-1] != 'extremely':
          next_word = True
      elif words[word_idx] == 'not' and idx > 0: # No repeat 'not'
        found_not = False
        for idx2 in range(1, min(4, idx+1)): # Up to 4 words back
          if phrase[idx-idx2]== 'not':
            found_not = True
        if not found_not:
          next_word = True
      elif words[word_idx] == 'not8' and idx > 0: # No repeat 'not8'
        found_not = False
        for idx2 in range(1, min(8, idx+1)): # Up to 8 words back
          if phrase[idx-idx2] == 'not8':
            found_not = True
        if not found_not:
          next_word = True
      elif words[word_idx] != '<EoS>':
        next_word = True
    phrase.append(words[word_idx])

  if the_padding:
    for _ in range(10):
      phrase.append('<noise>')

  # Adds padding at end for variable length phrases
  if variable_length:
    for i in range(phrase_length - length):
      phrase.append('<pad>')
  
  return phrase

def baseWordValues(toy_params):
  """ Generates the base word values which are used to score phrases """
  unordered = toy_params['unordered_class']

  if unordered:
    n_unordered = toy_params['n_unordered']
    corr_val = toy_params['corr_val'] if 'corr_val' in toy_params else 0.0
    base_word_vals = {}
    for i in range(n_unordered):
      base_word_vals['kinda'+str(i)] = np.zeros((n_unordered,))
      base_word_vals['kinda'+str(i)][i] = 1
      base_word_vals['very'+str(i)] = np.zeros((n_unordered,))
      base_word_vals['very'+str(i)][i] = 2
      base_word_vals['bad'+str(i)] = np.zeros((n_unordered,))
      base_word_vals['bad'+str(i)][i] = -1
      # base_word_vals['awful'+str(i)] = np.zeros((n_unordered,))
      # base_word_vals['awful'+str(i)][i] = -2
    # base_word_vals['kinda01'] = np.zeros((n_unordered,))
    # base_word_vals['kinda01'][0] = 1/2
    # base_word_vals['kinda01'][1] = 1/2
    # base_word_vals['bad0'] = np.zeros((n_unordered,))
    # base_word_vals['bad0'][0] = -1
    # base_word_vals['bad1'] = np.zeros((n_unordered,))
    # base_word_vals['bad1'][1] = -1
    # base_word_vals['kinda01'] = np.asarray([1,1,0])
    # base_word_vals['kinda12'] = np.asarray([0,1,1])
    # base_word_vals['kinda02'] = np.asarray([1,0,1])
    # base_word_vals['kinda03'] = np.asarray([1,0,0,1])
    # base_word_vals['kinda11110000'] = np.asarray([1,1,1,1,0,0,0,0])
    # base_word_vals['kinda00001111'] = np.asarray([0,0,0,0,1,1,1,1])
    # base_word_vals['kinda00110011'] = np.asarray([0,0,1,1,0,0,1,1])
    # base_word_vals['kinda11001100'] = np.asarray([1,1,0,0,1,1,0,0])
    # base_word_vals['kinda10101010'] = np.asarray([1,0,1,0,1,0,1,0])
    # base_word_vals['kinda01010101'] = np.asarray([0,1,0,1,0,1,0,1])
    # base_word_vals['kinda0'] = np.asarray([1, corr_val, 0])
    # base_word_vals['kinda1'] = np.asarray([0, 1 - corr_val, 0])
  else:
    base_word_vals = {'awful': -2.0, 'bad': -1.0, 'the': 0, 'good': 1.0, 'awesome': 2.0}

  return base_word_vals

def eval_toy_phrase(toy_phrase, toy_params):
  """ Evaluates a single toy phrase and returns a score """
  max_phrase_length = toy_params['phrase_length']
  unordered = toy_params['unordered_class']
  base_word_vals = toy_params['base_word_vals']
  corr_vals = toy_params['corr_vals'] if 'corr_vals' in toy_params else {}

  phrase_length = len(toy_phrase)
  extreme_length = 0 # range of influence of extreme
  not_length = 0 # range of influence of not

  if unordered:
    score = np.zeros((toy_params['n_unordered'],))
  else:
    score = 0

  for idx in range(phrase_length):
    if toy_phrase[idx] in list(base_word_vals.keys()):
      base_score = base_word_vals[toy_phrase[idx]]
      if not_length > 0: 
        base_score = -1 * base_score
      if extreme_length > 0: 
        base_score = 2 * base_score
      score += base_score
    elif toy_phrase[idx] == 'not':
      not_length = 4
    elif toy_phrase[idx] == 'not8':
      not_length = 8
    elif toy_phrase[idx] == 'extremely':
      extreme_length = 2
    # elif toy_phrase[idx] == '<EoS>': break
    # elif toy_phrase[idx] == '<pad>': break
    
    if not_length > 0: not_length -= 1 # decays not
    if extreme_length > 0: extreme_length -= 1 # decays extremely
  
  if corr_vals != {}: # Modifies scores if there are correlation values
    score = correlation_score_mod(score, toy_params)

  return score

def correlation_score_mod(score, toy_params):
  """ Modifies scores based on correlations """ 
  corr_vals = toy_params['corr_vals']
  n_classes = len(score)
  # print('Base score:', score)
  avgs = np.asarray([[1/2*(score0+score1) for score0 in score] for score1 in score])
  new_score = np.copy(score)

  for key in corr_vals:
    class0 = int(key[0])
    class1 = int(key[1])
    corr_val = corr_vals[key]
    avg_score = avgs[class0, class1]
    # print('Average is:', avg_score, 'Correlation value', corr_val)
    rand_coef = 2*np.random.randint(2)
    new_score[class0] = new_score[class0] - corr_val*score[class0] + rand_coef * corr_val * avg_score
    new_score[class1] = new_score[class1] - corr_val*score[class1] + (2 -rand_coef) * corr_val * avg_score
    # print('Updated score', new_score[0], new_score[1], new_score[2])
  
  return new_score

def classifySentiment(score, toy_params):
  """
  Turns a score of a toy phrase into a sentiment and corresponding tensor
  Contains definitions of thresholds for dividing into multiple classes
  """
  sentiments = toy_params['sentiments']
  unordered = toy_params['unordered_class']
  loss_type = toy_params['loss_type']
  phrase_length = toy_params['phrase_length']
  outer_prod = toy_params['outer_prod']
  eps2 = toy_params['eps2'] if 'eps2' in toy_params else 0.0
  eps5 = toy_params['eps5'] if 'eps5' in toy_params else 0.0

  n_sentiments = len(sentiments)
  if outer_prod:
    sentiment = []
    sentiment_tensor_final = np.zeros((n_sentiments,))
    for score_idx in range(score.shape[0]):
      if score[score_idx] >= 0:
        sentiment.append('Pos')
        sentiment_tensor_final[score_idx] = 1.0
      else:
        sentiment.append('Neg')
    # This is a hack that just sets all times of sentiment tensor equal to final
    sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
  elif unordered and eps2 > 0: # For matching onto 3-class Yelp
    if n_sentiments != 3:
      raise NotImplementedError('Not implemented beyond 3-class')
    
    sent2_thresh = phrase_length / (50/3)
    
    sentiment_tensor_final = np.zeros((n_sentiments,))
    if np.abs(score[0] - score[2]) < eps2 * sent2_thresh:
      sentiment = 'sent1'
      sentiment_tensor_final[1] = 1.0
    else:
      score[1] = (1-eps2) * score[1]
      sentiment = 'sent'+str(np.argmax(score))
      sentiment_tensor_final[np.argmax(score)] = 1.0
    # This is a hack that just sets all times of sentiment tensor equal to final
    sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
  elif unordered and eps5 > 0:  # For matching onto 5-class Yelp
    if n_sentiments != 5:
      raise NotImplementedError('Not implemented beyond 5-class')

    sent_thresh_a = phrase_length / (50/2)
    sent_thresh_b = sent_thresh_a + phrase_length / (50/4)
    
    sentiment_tensor_final = np.zeros((n_sentiments,))
    if np.abs(score[0] - score[4]) < eps5 * sent_thresh_a:
      sentiment = 'sent2'
      sentiment_tensor_final[2] = 1.0
    elif score[0] - score[4] < eps5 * sent_thresh_b and score[0] - score[4] > eps5 * sent_thresh_a:
      sentiment = 'sent1'
      sentiment_tensor_final[1] = 1.0
    elif score[0] - score[4] >  -1 *eps5 * sent_thresh_b and score[0] - score[4] < -1 * eps5 * sent_thresh_a:
      sentiment = 'sent3'
      sentiment_tensor_final[3] = 1.0
    else:
      score[1] = (1-eps5) * score[1]
      score[2] = (1-eps5) * score[2]
      score[3] = (1-eps5) * score[3]
      sentiment = 'sent'+str(np.argmax(score))
      sentiment_tensor_final[np.argmax(score)] = 1.0
    # This is a hack that just sets all times of sentiment tensor equal to final
    sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
  elif unordered:  # Usual unordered case
    sentiment = 'sent'+str(np.argmax(score))
    sentiment_tensor_final = np.zeros((n_sentiments,))
    sentiment_tensor_final[np.argmax(score)] = 1.0
    # This is a hack that just sets all times of sentiment tensor equal to final
    sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
  elif not unordered and toy_params['uniform_score']: # Ordered and uniform score
    
    five_order_match = toy_params['five_order_match'] if 'five_order_match' in toy_params else False

    if not five_order_match: # Standard case (not contrived five-class example)
      # Automatically subdivide possible scores based on number of classes
      sub_divs = []
      if len(toy_params['words']) == 3:
        score_range = 2*phrase_length + 1
        for class_idx in range(1, n_sentiments):
          sub_divs.append(-1*phrase_length + class_idx*score_range/n_sentiments)
      elif len(toy_params['words']) == 5:
        score_range = 4*phrase_length + 1
        for class_idx in range(1, n_sentiments):
          sub_divs.append(-2*phrase_length + class_idx*score_range/n_sentiments)

      class_found = False
      class_idx = 0
      while not class_found:
        if class_idx == n_sentiments - 1: # Checked all other classes
          class_found = True
        elif score <= sub_divs[class_idx]:
          class_found = True
        else:
          class_idx += 1
        
      sentiment = sentiments[class_idx]
    else: # Contrived five-class example, score is two-dimensional with (sentiment, neturality)
      sent_score = float(score[0])
      neutrality = float(score[1])

      if np.abs(neutrality) >= np.abs(sent_score) and neutrality > 0:
        sentiment = 'Three'
        # print('Three score', score)
      elif sent_score > 0 and neutrality > 0:
        sentiment = 'Four'
      elif sent_score > 0 and neutrality <= 0:
        sentiment = 'Five'
      elif sent_score < 0 and neutrality > 0:
        sentiment = 'Two'
      elif sent_score <= 0 and neutrality <= 0:
        sentiment = 'One'
      else:
        print('Score:', score)
        raise ValueError('Score outside of classification range')

    sentiment_tensor_final = np.zeros((n_sentiments,))
    sentiment_tensor_final[sentiments.index(sentiment)] = 1.0
    # This is a hack that just sets all times of sentiment tensor equal to final
    sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])

  else: # Ordered class examples (no uniform score)
  
    if toy_params['uniform_score']:
      # Used for 3-class sentiment analysis, about 1/3 will be neutral
      neutral_thresh = 1/3 * phrase_length

      # 5-class sentiment analysis, about 1/5 will be of each class
      three_star_thresh = 1/5 * phrase_length
      four_star_thresh = 3/5 * phrase_length
    else:
      neutral_thresh = 1/3 * phrase_length

      three_star_thresh = 0.05 * phrase_length
      four_star_thresh = 0.18 * phrase_length

    if loss_type == 'XE':
      if n_sentiments == 2:
        if score >= 0:
          sentiment = 'Good'
        else:
          sentiment = 'Bad'
      elif n_sentiments == 3:
        if score >= neutral_thresh:
          sentiment = 'Good'
        elif score <= -1 * neutral_thresh:
          sentiment = 'Bad'
        else:
          sentiment = 'Neutral'
      elif n_sentiments == 5:
        if score > four_star_thresh:
          sentiment = 'Five'
        elif score > three_star_thresh:
          sentiment = 'Four'
        elif score > -1 * three_star_thresh:
          sentiment = 'Three'
        elif score > -1 * four_star_thresh:
          sentiment = 'Two'
        else:
          sentiment = 'One'
      else:
        raise NotImplementedError('n_sentiments only implemented for 3 or 5 class')
      sentiment_tensor_final = np.zeros((n_sentiments,))
      sentiment_tensor_final[sentiments.index(sentiment)] = 1.0
      # This is a hack that just sets all times of sentiment tensor equal to final
      sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
    elif loss_type == 'MSE':
        sentiment = score
        sentiment_tensor = np.array([[0.0] if idx != phrase_length-1 else [score] for idx in range(phrase_length)])
  
  return sentiment, sentiment_tensor

def wordToIndex(word, word_bank):
  """ Converts a word into corresponding index in words """
  return word_bank.index(word)

def wordToTensor(word, word_bank):
  """ Turn a letter into a <1 x n_words> Tensor """
  n_words = len(word_bank)
  tensor = np.zeros((1, n_words))
  tensor[0][wordToIndex(word, word_bank)] = 1
  return np.array(tensor)

def phraseToTensor(phrase, word_bank):
  """ Turn a phrase into a <phrase_length x n_words> (an array of one-hot letter vectors) """
  n_words = len(word_bank)
  tensor = np.zeros((len(phrase), n_words))
  for wi, word in enumerate(phrase):
    if word not in ['<pad>', '<noise>', '<avg>']:
      tensor[wi][wordToIndex(word, word_bank)] = 1
    elif word in ['<avg>']: # returns average of all other words
      tensor[wi] = 1/(n_words - 1) * np.ones((n_words,))
      tensor[wi][wordToIndex(word, word_bank)] = 0
  return np.array(tensor)

def tensorToPhrase(tensor, word_bank):
  """ Turn an array of one-hot letter vectors into a phrase """
  phrase = []
  for idx in range(tensor.shape[0]):
      hot_idx = np.argmax(tensor[idx])
      phrase.append(word_bank[hot_idx])
  return phrase

def randomTrainingExample(toy_params, ):
  """
  Generates a random training example consisting of phrase and sentiment and corresponding tensors

  Returns:
  sentiment_tensor: time x input dim (word bank size)
  sentiment_tensor: 1 x output dim (sentiment bank size)
  target_mask:

  """
  # Unpacks toy_params (with defaults)
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']
  sentiments = toy_params['sentiments']
  loss_type = toy_params['loss_type'] if 'loss_type' in toy_params else 'XE'
  variable_length = toy_params['variable_length'] if 'variable_length' in toy_params else 'False'
  min_phrase_len = toy_params['min_phrase_len'] if 'min_phrase_len' in toy_params else 10
  unordered_class = toy_params['unordered_class'] if 'unordered_class' in toy_params else 'False'
  uniform_score = toy_params['uniform_score'] if 'uniform_score' in toy_params else False
  filter = toy_params['filter'] if 'filter' in toy_params else 0
  
  if uniform_score and unordered_class: # Uniform score generation for unordered classes
    n_scores_phrase = n_scores(len(sentiments), phrase_length)
    score_idx = np.random.randint(n_scores_phrase)
    score = index_to_score_unordered(score_idx, len(sentiments), phrase_length)
    phrase = score_to_phrase_unordered(score, toy_params)

    sentiment, sentiment_tensor = classifySentiment(score, toy_params)
  elif uniform_score and not unordered_class: # Uniform score generation for ordered classes
    if len(words) == 3: # good, bad, the case
      n_scores_phrase = 2*phrase_length+1
      score_idx = np.random.randint(n_scores_phrase)
      score = index_to_score_ordered(score_idx, phrase_length)
      phrase = score_to_phrase_ordered(score, toy_params)
    else: # more general case
      score_to_idx_map = toy_params['score_to_idx_map']
      score_vals = list(score_to_idx_map.keys())
      score_idx = np.random.randint(len(score_to_idx_map)) # random score index
      score = score_vals[score_idx]
      phrase = score_to_phrase_ordered_general(score, score_to_idx_map, toy_params)

    sentiment, sentiment_tensor = classifySentiment(score, toy_params)
  elif filter > 0 and not uniform_score:
    if toy_params['n_unordered'] != 3:
      raise NotImplementedError('Filters not yet implemented for n_sents != 3')
    valid_phrase = False
    while not valid_phrase:
      phrase = make_toy_phrase(toy_params)
      score = eval_toy_phrase(phrase, toy_params)
      if score[0] <= (phrase_length/2 - filter) or score[2] <= (phrase_length/2 - filter):
        valid_phrase = True  
      else:
        min_sub = np.min([score[0] - np.max([score[0]-filter, 0]), score[2] - np.max([score[2]-filter, 0])])
        score[0] -= min_sub
        score[2] -= min_sub
        score[1] += 2*min_sub
        valid_phrase = True
          
    sentiment, sentiment_tensor = classifySentiment(score, toy_params)
  else:
    phrase = make_toy_phrase(toy_params)
    if toy_params['extreme_test']: # special case for extremeness Yelp test
      sentiment, sentiment_tensor = classifySentimentExtreme(phrase, toy_params)
    else:
      score = eval_toy_phrase(phrase, toy_params)
      sentiment, sentiment_tensor = classifySentiment(score, toy_params)
  
  targets_t = np.zeros(phrase_length-1)
  phrase_tensor = phraseToTensor(phrase, words)
  
  length = len(phrase)
  if variable_length:
    while phrase[length-1] == '<pad>':
      length -= 1
  target_mask = np.array([length-1], dtype=int) # When target is defined.
  
  return sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask

def generate_data(dataset_size, toy_params, out_size, auto_balance=False, jnp_arrays=True):
  """
  Generate training data in numpy and then converts to JAX arrays. Also implements filters if being used
  """
  uniform_score = toy_params['uniform_score'] if 'uniform_score' in toy_params else False
  filter_classes = toy_params['filter_classes'] if 'filter_classes' in toy_params else False

  syn_sentiments = []
  syn_phrases = []
  syn_targets_np = np.zeros((dataset_size, toy_params['phrase_length'], out_size))
  syn_inputs_np = np.zeros((dataset_size, toy_params['phrase_length'], len(toy_params['words'])))
  syn_target_masks_np = np.zeros((dataset_size, 1))
  # syn_target_masks_np = []

  outer_prod = toy_params['outer_prod'] if 'outer_prod' in toy_params else False

  if uniform_score and toy_params['unordered_class']:
    check_word_bank_for_uniform_score(toy_params['words'], len(toy_params['sentiments']))
  elif uniform_score and not toy_params['unordered_class']:
    check_word_bank_for_uniform_score_ordered(toy_params['words'])

    if len(toy_params['words']) == 5 or len(toy_params['words']) == 6:
      if 'score_to_idx_map' not in toy_params:
        score_vals = toy_params['score_vals'] if 'score_vals' in toy_params else [-2, -1, 1, 2]
        toy_params['score_to_idx_map'] = generate_score_map(len(score_vals), toy_params['phrase_length'], score_vals)


  start_time = time.time()
  if auto_balance:
    if outer_prod:
      n_classes = 2**len(toy_params['sentiments'])
    else:
      n_classes = len(toy_params['sentiments'])
    if filter_classes:
      class_filter_idxs = []
      print('Filtering out classes:', class_filter_idxs)
      num_per_class = np.ones((n_classes,))*int(dataset_size/(n_classes- len(class_filter_idxs)))
      if class_filter_idxs != []:
        num_per_class[tuple(class_filter_idxs)] = 0
      # Special code to test imbalanced datasets
      # num_per_class =  np.ones((n_classes,))*int(99/300 * dataset_size)
      # num_per_class[[3]] = int(3/300 * dataset_size)
      # num_per_class = np.ones((n_classes,))*int(39/1000 *  dataset_size)
      # num_per_class[[0]] = int(920/1000 * dataset_size)
      # num_per_class[[3]] = int(2/1000 * dataset_size)
      # num_per_class[[class_filter_idxs]] = 3/100 * int(dataset_size/3)
      # num_per_class[[3, 5, 6]] = 1/100*int(dataset_size/3)
    else:
      num_per_class = np.ones((n_classes,))*int(dataset_size/n_classes)
    print('Looking for num per class:', num_per_class)
    class_count = np.zeros((n_classes,))
    complete = False
    while not complete:
      sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask = randomTrainingExample(toy_params)
      if outer_prod: # Treats sentiment like a binary number and converts it into a class index
        class_idx = int(np.sum([sentiment_tensor[0, sent_idx]*(2**sent_idx) for sent_idx in range(sentiment_tensor.shape[1])]))
      else:
        class_idx = np.argmax(sentiment_tensor[0])

      if class_count[class_idx] < num_per_class[class_idx]:
        trial = int(np.sum(class_count))
        class_count[class_idx] += 1

        syn_targets_np[trial, :, :] = sentiment_tensor
        syn_inputs_np[trial, :, :] = phrase_tensor
        if len(target_mask) > 1:
          raise NotImplementedError
        syn_target_masks_np[trial, :] = target_mask

        if (class_count == num_per_class).all(): # Checks to see if finished
          complete = True
  else:
    for trial in range(dataset_size):
      sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask = randomTrainingExample(toy_params)
      
      syn_targets_np[trial, :, :] = sentiment_tensor
      syn_inputs_np[trial, :, :] = phrase_tensor
      if len(target_mask) > 1:
        raise NotImplementedError
      syn_target_masks_np[trial, :] = target_mask

  print('Sythentic data generated in: {:0.2f} sec. Autobalanced: {}. Uniform score: {}'.format(
      time.time() - start_time, auto_balance, uniform_score))

  if jnp_arrays:   # Converts to JAX arrays
    syn_targets = jnp.asarray(syn_targets_np)
    syn_inputs = jnp.asarray(syn_inputs_np)  
    syn_target_masks =  jnp.asarray(syn_target_masks_np, dtype=jnp.int32)

    syn_data = {
      'inputs': syn_inputs,  # Phrase tensors: dataset_size x phrase_len x in_dim
      'labels': syn_targets, # Sentiment tensors: dataset_size x phrase_len x out_dim
      'index': syn_target_masks, # Target mask: list of integers up to phrase_len
    }
  else:
    syn_data = {
      'inputs': syn_inputs_np,  # Phrase tensors: dataset_size x phrase_len x in_dim
      'labels': syn_targets_np, # Sentiment tensors: dataset_size x phrase_len x out_dim
      'index': syn_target_masks_np, # Target mask: list of integers up to phrase_len
    }

  return syn_data

#######################################################################################
############### Functions having to do with uniform score distributions ############### 
#######################################################################################

def enumerate_phrases(toy_params):
  """ Enumerates all possible phrases """
  
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']

  total_phrases = len(words)**phrase_length

  score_dict = {}
  scores = []
  for phrase_idx in range(total_phrases):
    phrase = index_to_phrase(phrase_idx, toy_params)
    score = eval_toy_phrase(phrase, toy_params)
    if score in score_dict:
      score_dict[score].append(phrase_idx)
    else:
      score_dict[score] = [phrase_idx]

  return score_dict

def index_to_phrase(index, toy_params):
  """ Converts an index to a phrase """
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']

  phrase = ['' for _ in range(phrase_length)]
  for idx in range(phrase_length):
    index, rem = divmod(index, len(words))
    phrase[idx] = words[rem]
    
  return phrase

def score_to_phrase_unordered(score, toy_params):
  """ Converts a score to a phrase, only works for certain word collections """
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']

  phrase = []

  for score_idx in range(len(score)):
    phrase.extend(['kinda'+str(score_idx) for _ in range(int(score[score_idx]))])
  phrase.extend(['the' for _ in range(phrase_length - len(phrase))])

  np.random.shuffle(phrase)
    
  return phrase

def enumerate_scores(toy_params):
  """ Converts a score to a phrase, only works for certain word collections """
  phrase_length = toy_params['phrase_length']
  n_unordered = toy_params['n_unordered']
  # filter = toy_params['filter'] if 'filter' in toy_params else 0

  scores = []
  if n_unordered == 2:
    for score1 in range(phrase_length+1):
      scores.append([score1, phrase_length - score1])
    scores_filtered = scores
  if n_unordered == 3:
    for score1 in range(phrase_length+1):
      for score2 in range(phrase_length - score1+1):
          scores.append([score1, score2, phrase_length - score1 - score2])

    scores_filtered = []
    for score_idx in range(len(scores)):
      score = scores[score_idx]
      if score[0] <= (phrase_length/2 - filter) or score[1] <= (phrase_length/2 - filter):
        scores_filtered.append(score)
  else:
    raise NotImplementedError()

  return scores_filtered

def check_word_bank_for_uniform_score(words, n_classes):
  """ 
  Checks if current word bank can be used for uniform score generation.
  Assumes scores are of the form {'kinda0', 'kinda1', ..., 'kinda(n_classes-1)', 'the'}
  """
  contradiction = False
  for i in range(n_classes):
    if 'kinda'+str(i) not in words:
      contradiction = True
  if 'the' not in words:
    contradiction = True
  if len(words) != n_classes+1:
    contradiction = True
  if contradiction:
    raise ValueError('Word bank incompatible with uniform scores.')

def n_scores(n_classes, max):
  """ 
  Determines the number of possible scores for n_classes and a phrase length of max.
  Assumes scores are of the form {'kinda0', 'kinda1', ..., 'kinda(n_classes-1)', 'the'}
  """
  return np.prod([n + 1 + max for n in range(0,n_classes)])/np.prod([n + 1 for n in range(n_classes)])

def index_to_score_unordered(index, n_classes, max_val):
  """ 
  Map an index to a score, used to generate scores with uniform probability.
  Assumes scores are of the form {'kinda0', 'kinda1', ..., 'kinda(n_classes-1)', 'the'}
  """
  if index >= n_scores(n_classes, max_val):
    raise ValueError('Index is too large!')
  score = np.zeros((n_classes,))
  max = max_val
  n = n_classes - 1
  for score_idx in range(n_classes):
    if score_idx == n_classes - 1:
      score[score_idx] = index
    else:
      current_val = 0
      val_found = False
      while not val_found:
        if index < n_scores(n, max):
          val_found = True
          score[score_idx] = current_val
          n -= 1
        else:
          index -= n_scores(n, max)
          max -= 1
          current_val += 1

  return score

def check_word_bank_for_uniform_score_ordered(words):
  """ 
  Checks if current word bank can be used for uniform score generation ordered.
  Assumes scores are of the form {'good', 'bad', 'the'}
  """
  contradiction = False
  
  if len(words) == 3:
    if 'good' not in words:
      contradiction = True
    if 'bad' not in words:
      contradiction = True
    if 'the' not in words:
      contradiction = True
  elif len(words) == 5:
    if words != ['awesome', 'good', 'bad', 'awful', 'the']: # Needs this exact ordering for now
      contradiction = True
  elif len(words) == 6:
    if words != ['awesome', 'good', 'bad', 'awful', 'okay', 'the']: # Needs this exact ordering for now
      contradiction = True
  else:
    contradiction = True
  if contradiction:
    raise ValueError('Word bank incompatible with uniform scores.')

def index_to_score_ordered(index, max_val):
  """ 
  Map an index to a score, used to generate scores with uniform probability for ordered classes.
  """
  if index >= (2*max_val+1):
    raise ValueError('Index is too large!')
  return index - max_val

def score_to_phrase_ordered(score, toy_params):
  """ Converts a score to a phrase, only works for certain word collections """
  phrase_length = toy_params['phrase_length']

  phrase = []
  if score > 0:
    phrase.extend(['good' for _ in range(int(score))])
  elif score < 0:
    phrase.extend(['bad' for _ in range(int(np.abs(score)))])
  if score % 2 == 1: # Odd scores get a 'the'
    phrase.extend(['the'])

  # Randomly determines the number of 'the' or 'good/bad' occurences
  max_zero_sums = int((phrase_length - np.abs(score)) // 2)
  n_the_pairs = np.random.randint(max_zero_sums+1)
  
  for _ in range(n_the_pairs):
    phrase.extend(['the', 'the'])
  for _ in range(max_zero_sums-n_the_pairs):
    phrase.extend(['good', 'bad'])

  np.random.shuffle(phrase)
    
  return phrase

def generate_score_map(n_classes, length, score_vals):
  """ Used fo uniform score distribution of ordered datasets with more words than just good, bad, the """
  print('Generating score map for uniform scores...')
  total_scores = int(n_scores(n_classes, length))
  score_map = {}
  for score_idx in range(total_scores):
    raw_count = index_to_score_unordered(score_idx, n_classes, length)
    # print('Raw count', raw_count)
    # print('score vals', score_vals)
    score = tuple(np.dot(raw_count, score_vals))
    if score in score_map:
      score_map[score].append(score_idx)
    else:
      score_map[score] = [score_idx]
  return score_map

def score_to_phrase_ordered_general(score, score_map, toy_params):
  """ Converts a score to a phrase, only works for certain word collections """
  phrase_length = toy_params['phrase_length']
  words = toy_params['words']

  score = tuple(score) # converts the score to a for indexing score_map
  phrase = []
  # Random index for the given score
  phrase_idx = np.random.randint(len(score_map[score]))
  # Raw count of words that are not 'the'
  raw_count = index_to_score_unordered(score_map[score][phrase_idx], len(words)-1, phrase_length)

  for count_idx in range(len(raw_count)):
    phrase.extend([words[count_idx] for _ in range(int(raw_count[count_idx]))])
  phrase.extend(['the' for _ in range(phrase_length - len(phrase))])

  np.random.shuffle(phrase)
    
  return phrase

# Matching to extremeness Yelp data
def classifySentimentExtreme(toy_phrase, toy_params):
  """
  Turns a toy phrase into a score and then into a sentiment and corresponding tensor
  Contains definitions of thresholds for dividing into multiple classes
  """
  sentiments = toy_params['sentiments']
  unordered = toy_params['unordered_class']
  loss_type = toy_params['loss_type']
  phrase_length = toy_params['phrase_length']

  n_sentiments = len(sentiments)

  if unordered or loss_type == 'MSE':
    raise NotImplementedError('Extremeness not implemented for unordered')
  
  # Used for 3-class sentiment analysis, about 1/3 will be neutral
  neutral_thresh = 4
  extreme_threshold = 50/4 + np.sqrt(3/16 * 50) - 2

  score = np.zeros((2,)) # first score for sentiment, second score for extremeness
  base_word_vals = {'bad': np.asarray([-1, 0.0]), 'the': np.asarray([0.0, 0.0]), 'good': np.asarray([1, 0.0])}
  extreme_length = 0 # range of influence of extreme
  not_length = 0 # range of influence of not

  for idx in range(phrase_length):
    if toy_phrase[idx] in list(base_word_vals.keys()):
      base_score = base_word_vals[toy_phrase[idx]]
      if not_length > 0: 
        base_score = -1 * base_score
      if extreme_length > 0: 
        base_score = 2 * base_score
      score += base_score
    elif toy_phrase[idx] == 'not':
      not_length = 4
    elif toy_phrase[idx] == 'extremely':
      extreme_length = 2
      score += np.asarray([0.0, 1.0]) # Adds to extremeness score
    elif toy_phrase[idx] == '<EoS>': break
    
    if not_length > 0: not_length -= 1 # decays not
    if extreme_length > 0: extreme_length -= 1 # decays extremely

  if score[1] < extreme_threshold: # Not an extreme score
    if score[0] >= neutral_thresh:
      sentiment = 'Good'
    elif score[0] <= -1 * neutral_thresh:
      sentiment = 'Bad'
    else:
      sentiment = 'Neutral'
  else:
    if score[0] >= 0:
      sentiment = 'Good'
    else:
      sentiment = 'Bad'
  
  sentiment_tensor_final = np.zeros((n_sentiments,))
  sentiment_tensor_final[sentiments.index(sentiment)] = 1.0
  sentiment_tensor = np.array([np.zeros((n_sentiments,)) if idx != phrase_length-1 else sentiment_tensor_final for idx in range(phrase_length)])

  return sentiment, sentiment_tensor
