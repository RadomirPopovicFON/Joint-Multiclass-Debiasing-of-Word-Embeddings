from numpy.linalg import svd
import numpy as np
import itertools
import math
import random
import copy

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def neutralize_vectors(vectors, v):
    '''
    Neutralizing vectors (Deducting from vectors their vector projection to a particular vector v-used as a centroid)

    Parameters
    ----------
    vectors: ndarray | Vectors for neutralizing
    v: ndarray | Vector from which neutralization will occur - centroid

    Returns
    -------
    neutralized_vectors: ndarray | Neutralized vectors
    '''
    v_part = (np.transpose(v / np.dot(v, v))).reshape(-1, 1)
    vectors_part = np.dot(vectors, np.transpose([v]))
    return vectors - np.dot(vectors_part, np.transpose(v_part))


def make_vectors_equidistant(center_vector, dic_of_vectors, r):
    '''
    Equidistancing step from HardWEAT

    Parameters
    ----------
    center_vector: ndarray | Central vector
    dic_of_vectors: dict | Vectors to be changed, where key is word and value is vector representation
    radius: int | radius of the circle

    Returns
    -------
    dict_of_vectors: dict | containing words as keys, and equidistanced positions as values
    '''
    n = len(dic_of_vectors)
    dimension = len(center_vector)
    null = nullspace(center_vector)
    dict_of_vectors = copy.deepcopy(dic_of_vectors)

    v1, v2 = 0, 0
    orthogonal_vectors_combinations = [[np.transpose(null[:, i]), np.transpose(null[:, j])] for i in
                                       range(0, dimension - 1) for j in range(0, dimension - 1) if (i != j)]

    for i in range(0, 100):
        v1, v2 = orthogonal_vectors_combinations[random.randint(0, len(orthogonal_vectors_combinations) - 1)]
        if (float("%.10f" % np.dot(v1, v2)) == 0.0):
            break
        if (i == 99):
            raise ValueError("Cannot find orthogonal vectors!")

    for i, (key, value) in enumerate(dict_of_vectors.items()):
        dict_of_vectors[key] = center_vector + r * math.cos((2 * math.pi * i) / n) * v1 + r * math.sin(
            (2 * math.pi * i) / n) * v2

    del dic_of_vectors
    return dict_of_vectors


def nullspace(A):
    """
    This method returns nullspace of vector, using SVD.
    Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html

    Parameters
    ----------
    A: ndarray | Matrix for which nullspace shall be found

    Returns
    -------
    nullspace_matrix: ndarray | Nullspace
    """
    atol = 1e-50
    rtol = 0

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def get_SW_dict(bias_weat, d_threshold=1):
    """
    This method returns target attribute dictionary from WEAT scores in format necessary for SoftWEAT

    Parameters
    ----------
    bias_weat: dict | WEAT scores where key represents bias class that contains list of sextuples - ['target set 1', 'target set 2',
    'attribute set 1', 'attribute set 2', 'effect size', 'p value']
    d_threshold: float | Threshold that separates biased scores from WEAT experiments

    Returns
    -------
    nullspace_matrix: ndarray | Nullspace
    """
    target_at_dict = {}

    for class_name in ['gender', 'race', 'religion']:
        class_results = bias_weat[class_name]
        for res in class_results:

            test = res
            d_value = round(res[-1], 2)
            # Revert order of negative effect sizes
            if (d_value < 0): temp = test[2]; test[2] = test[3]; test[3] = temp; test[-1] = np.abs(test[-1])

            if (np.abs(d_value) > d_threshold):
                for i, subclass_name in enumerate([test[0], test[1]]):
                    if subclass_name not in target_at_dict:
                        target_at_dict[subclass_name] = set()
                    target_at_dict[subclass_name].add(test[i + 2])
            # Each target set of words will receive its corresponding attribute sets of words with which it forms bias

    return target_at_dict


def make_translation_matrix(T, l=1):
    """
    For SoftWEAT purposes

    Parameters
    ----------
    T: list | vector translation values
    l: float | level of debiasing

    Returns
    -------
    T: ndarray | Translation matrix
    """
    matrix = np.eye((len(T) + 1))
    for t_i, value in enumerate(T):
        matrix[t_i, -1] = value * l
    return np.array(matrix)


def union_of_dictionaries(*dicts):
    """
    Concatenate dictionaries

    Parameters
    ----------
    *dicts: list of dict | List for which concatenation shall occur

    Returns
    -------
    concatenated_dict: dict | Concatenated dictionary
    """
    return dict(itertools.chain.from_iterable(dct.items() for dct in dicts))


# ----------------------------------------------------------------- #
# * Following methods are used for Sentiment Analysis Experiments * #
# ----------------------------------------------------------------- #


def get_dataset_and_dicts(targets_sets, params):
    """
    Processing IMDB Dataset for Sentiment Analysis task.

    Parameters
    ----------
    targets_sets: String | Key in form '<target set 1>_<target set 2>', e.g. islam_christianity for which we assess variance within S.A. task
    params: dict | parameters_dict from sets.get_sent_analysis_sets() method

    Returns
    -------
    datasets: dict | training, test, label data (both padded and non-padded, mappings between words and indices)
    """

    df = pd.read_csv('Datasets/IMDB_Dataset.csv')
    df = df.reindex(np.random.permutation(df.index))

    X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment, shuffle=False)
    tk = Tokenizer(num_words=params["NB_WORDS"])
    tk.fit_on_texts(df.review)
    x_train = tk.texts_to_sequences(X_train)
    x_test = tk.texts_to_sequences(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Filtering input to satisfy learning constraints
    targets_sets = filter_emb(targets_sets, tk.word_index)
    x_train, y_train = null_intersection(x_train, y_train,
                                         set([word for set_name in targets_sets for word in targets_sets[set_name]]),
                                         tk.word_index)

    x_train_copy, x_test_copy, y_train_copy, y_test_copy = [], [], [], []

    for i in range(0, len(x_train)):
        if len(x_train[i]) < params['MAX_LEN']:
            x_train_copy.append(x_train[i])
            y_train_copy.append(y_train[i])

    for i in range(0, len(x_test)):
        if len(x_test[i]) < params['MAX_LEN']:
            x_test_copy.append(x_test[i])
            y_test_copy.append(y_test[i])

    x_train_padded = pad_sequences(x_train_copy, maxlen=params['MAX_LEN'], padding='post')
    x_test_padded = pad_sequences(x_test_copy, maxlen=params['MAX_LEN'], padding='post')
    print(f'Training shape: {x_train_padded.shape}\nTest shape: {x_test_padded.shape}')

    datasets = {'x_train': np.array(x_train_copy), 'x_test': np.array(x_test_copy),
                'x_train_padded': np.array(x_train_padded), 'x_test_padded': np.array(x_test_padded),
                'y_train': np.array(y_train_copy), 'y_test': np.array(y_test_copy),
                'word_2_index': tk.word_index, 'index_2_word': tk.index_word, 'targets_sets': targets_sets}

    return datasets


def change_the_sentences(datasets, set_name, sorting_top_thresh=100):
    """
    Add to the end of the sentence random word from target set

    Parameters
    ----------
    datasets: dict | Dictionary from get_dataset_and_dicts_method()
    set_name: String | 'first_set' or 'second_set'
    sorting_top_thresh: int | Number of sorted shortest sentences we're taking as a test input

    Returns
    -------
    x_input_new: ndarray | Modified test input - training
    y_input_new: ndarray | Modified test input - label
    """
    x_input, x_input_padded = datasets['x_test'], datasets['x_test_padded']
    y_input = datasets['y_test']
    word_to_id = datasets['word_2_index']
    t_set = datasets['targets_sets'][set_name]

    x_input_new, y_input_new = [], []
    x_input_padded_new = copy.deepcopy(x_input_padded)
    sentences_length_order = [len(x_input[i]) for i in range(0, len(x_input))]

    for k, j in enumerate(np.array(sentences_length_order).argsort()[:sorting_top_thresh]):
        index_of_last_nonzero_element = max(np.nonzero(x_input_padded[j])[0])
        x_input_padded_new[j][index_of_last_nonzero_element + 1:index_of_last_nonzero_element + 2] = [
            word_to_id[t_set[random.randrange(len(t_set))]]]
        x_input_new.append(x_input_padded_new[j])
        y_input_new.append(y_input[j])

    print(f'Shape of modified test input: {np.array(x_input_new).shape}')

    return np.array(x_input_new), np.array(y_input_new)


def filter_emb(target_sets, word_index):
    """
    Filtering opposite target sets to be equal size and using only ones in embedding

    Parameters
    ----------
    target_sets: dict | target_sets_dict from sets.get_sent_analysis_sets() method
    word_index: dict | Word as a key and index as value

    Returns
    -------
    filtered_word_index: dict | Two opposing sets of words
    """
    lengths = [0, 0]
    for i, key_idx in enumerate(list(target_sets.keys())):
        target_sets[key_idx] = [word for word in target_sets[key_idx] if word in word_index]
        lengths[i] = len(target_sets[key_idx])

    return {key_idx: word_list[0:min(lengths)] for key_idx, word_list in target_sets.items()}


def null_intersection(x_train, y_train, target_words, word_index):
    """
    Eliminating training set that contains target words

    Parameters
    ----------
    x_train: ndarray | Training data
    y_train: ndarray | Label data
    target_words: list | List of target words to exclude from learning procedure
    word_index: dict | Word to index mapping

    Returns
    -------
    x_train_new: ndarray | Training data
    y_train_new: ndarray | Label data
    """
    x_train_new, y_train_new = [], []

    for i, x_t in enumerate(x_train):
        intersection = len(set(x_t).intersection([word_index[word] for word in target_words]))
        if intersection == 0:
            x_train_new.append(x_t)
            y_train_new.append(y_train[i])

    return x_train_new, y_train_new