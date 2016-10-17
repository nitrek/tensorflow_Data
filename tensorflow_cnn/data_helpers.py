import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    p_examples = list(open("/data/public.txt", "r").readlines())
    p_examples = [s.strip() for s in p_examples]
    i_examples = list(open("/data/internal.txt", "r").readlines())
    i_examples = [s.strip() for s in i_examples]
	r_examples = list(open("/data/restricted.txt", "r").readlines())
    r_examples = [s.strip() for s in r_examples]
	h_examples = list(open("/data/highly.txt", "r").readlines())
    h_examples = [s.strip() for s in h_examples]
	
	
	
    # Split by words and clean the data
    x_text = p_examples + i_examples + r_examples + h_examples 
    x_text = [clean_str(sent) for sent in x_text]
	
    # Generate labels
    p_labels = [[0, 0, 0, 1] for _ in p_examples]
    i_labels = [[0, 0, 1, 0] for _ in i_examples]
	r_labels = [[0, 1, 0, 0] for _ in r_examples]
    h_labels = [[1, 0, 0, 0] for _ in h_examples]
	
    y = np.concatenate([p_labels, i_labels, r_labels , h_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

