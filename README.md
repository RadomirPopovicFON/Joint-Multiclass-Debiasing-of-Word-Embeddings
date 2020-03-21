# Joint-Multiclass-Debiasing-of-Word-Embeddings
This repository contains code for the paper: Joint Multiclass Debiasing techniques HardWEAT and SoftWEAT, accepted for 25th International Symposium on Intelligent Systems (ISMIS 2020), Graz, Austria, September 2020. 

*URL of the paper will appear here*

Here, small samples of [Word2Vec](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM), [GloVe](https://nlp.stanford.edu/projects/glove/), [FastText](https://fasttext.cc/docs/en/english-vectors.html) Embeddings are used, with words [having minimum occurrence of 200 in English Wikipedia](https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt), along with [highly polarizing IMDB Movie Dataset](https://www.aclweb.org/anthology/P11-1015/). On these datasets, HardWEAT and SoftWEAT are examined via WEAT bias experiments, Mikolov analogy, Rank Similarity and Sentiment Analysis tasks (see [Main.ipynb](https://github.com/RadomirPopovicFON/Joint-Multiclass-Debiasing-of-Word-Embeddings/blob/master/Main.ipynb)). Furthermore, corresponding [online appendix](https://github.com/RadomirPopovicFON/Joint-Multiclass-Debiasing-of-Word-Embeddings/blob/master/Online%20Appendix.pdf) for the paper is provided.

## Requirements
- numpy
- sklearn
- keras
- tensorflow
- gensim
- seaborn
- matplotlib
- scipy
- nltk
- copy
- pickle
- time
- itertools
- math
- random
- pandas


