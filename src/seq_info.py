from collections import Counter

def words_counter(text_column: list) -> dict:
  """
  Counting unique words occurences

  Parameters:
  -----------
  X_train - training data
  Returns:
  --------
  word_count - dict with number of unique word occurences in training sentences
  """
  count = Counter()
  for text in text_column.values:
    for word in text:
      count[word] += 1
    
  return count

def get_mean_seq_length(list_seq: list) -> float:
  """
  Random upsampling minor class

  Parameters:
  -----------
  X_train - training data
  y_train - training labels
  Returns:
  --------
  X_res - resampled training data
  y_res - resampled training labels
  """
  length = 0
  for seq in list_seq:
    length += len(seq)
  
  return length/len(list_seq)