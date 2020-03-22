# Joint-Multiclass-Debiasing-of-Word-Embeddings
This repository contains code for the paper: *Joint Multiclass Debiasing of Word Embeddings*, accepted for *25th International Symposium on Intelligent Systems (ISMIS 2020), Graz, Austria, September 2020*. 

*URL of the paper will appear here*

## Description

*Word Embedding, as an important tool for numerous downstream NLP tasks, can contain different kinds of biases, based on gender, religion, race. In this direction, by extending work from [Bolukbasi et al.](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) and [Caliskan et al.](https://science.sciencemag.org/content/356/6334/183) HardWEAT and SoftWEAT are created with an aim to reduce this phenomenon simultaneously/jointly on multiple bias classes/categories. Former completely eliminates bias measured with WEAT, while latter provides an user with a choice to what extent debiasing procedure will occur. Experiments show that the two methods are able to both decrease bias levels while minimizing the structure modification of vector representation. In addition, debiasing of Word Embeddings, translates to variance decline of polarity scores within the task of Sentiment Analysis.*

![SoftWEAT](https://github.com/RadomirPopovicFON/Joint-Multiclass-Debiasing-of-Word-Embeddings/blob/master/Images/softweat_change.png "SoftWEAT Debiasing on FastText Word Embedding.")

Here, samples of [Word2Vec](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM), [GloVe](https://nlp.stanford.edu/projects/glove/), [FastText](https://fasttext.cc/docs/en/english-vectors.html) Embeddings are used, with words [having minimum occurrence of 200 in English Wikipedia](https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt), along with [highly polarizing IMDB Movie Dataset](https://www.aclweb.org/anthology/P11-1015/). On these datasets, HardWEAT and SoftWEAT are examined via WEAT bias experiments, Mikolov analogy, Rank Similarity and Sentiment Analysis tasks (see [Main.ipynb](https://github.com/RadomirPopovicFON/Joint-Multiclass-Debiasing-of-Word-Embeddings/blob/master/Main.ipynb)). Furthermore, corresponding [online appendix](https://github.com/RadomirPopovicFON/Joint-Multiclass-Debiasing-of-Word-Embeddings/blob/master/Online%20Appendix.pdf) for the paper is provided.

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
