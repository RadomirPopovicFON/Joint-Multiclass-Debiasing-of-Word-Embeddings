import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from gensim.models.keyedvectors import Word2VecKeyedVectors as GensimPairs
from gensim.test.utils import datapath
from scipy.stats import spearmanr

EVAL_DATASETS = 'Datasets/eval_datasets/'

def emb_to_gensim(e):
	'''
	Convert embedding to gensim format

	Parameters
	----------
	e: Embedding | instance of class Embedding

	Returns
	-------
	gpairs: Word2VecKeyedVectors | Embedding in Gensim format
	'''
	rank = np.shape(e.vectors)[1]
	gpairs = GensimPairs(rank)
	gpairs.add([word for word in e.words], [np.array(v) for v in e.vectors])
	return gpairs

def mikolov_evaluation(gpairs):
	'''
	Evaluate Embeddings via Mikolov task

	Parameters
	----------
	gpairs: Word2VecKeyedVectors | Embedding in Gensim format

	Returns
	-------
	dict_res: dict | Mikolov analogy task results
	'''

	results = gpairs.wv.evaluate_word_analogies(datapath('questions-words.txt'))

	dict_res = {}
	for i in range(0, 15):
		section_name = results[1][i]['section']
		result = round(len(results[1][i]['correct'])/(len(results[1][i]['correct'])+len(results[1][i]['incorrect'])), 2)
		dict_res[section_name] = result

	return dict_res

def rank_evaluation(e):
	'''
	Evaluating embeddings using rank evaluation task (Using the same logic/datasets from Conceptor Debiasing paper)
	Paper reference: https://www.aclweb.org/anthology/W19-3806.pdf
	Code source: https://github.com/jsedoc/ConceptorDebias
	Datasets source: https://github.com/mfaruqui/eval-word-vectors/tree/master/data/word-sim

	Parameters
	----------
	e: Embedding | instance of class Embedding

	Returns
	-------
	results: dict | Per each dataset generate result
	'''
	results = {}
	datasets = {'WS':(EVAL_DATASETS+'EN-WS-353-ALL.txt', '\t' ), 'RG65':(EVAL_DATASETS+'EN-RG-65.txt', '\t'),
			'RW':(EVAL_DATASETS+'EN-RW-STANFORD.txt', '\t' ), 'MEN':(EVAL_DATASETS+'EN-MEN-TR-3k.txt', ' '),
			'MTurk':(EVAL_DATASETS+'EN-MTurk-771.txt', ' ' ), 'SimLex':(EVAL_DATASETS+'EN-SIMLEX-999.txt', '\t' ),
			'SimVerb':(EVAL_DATASETS+'EN-SimVerb-3500.txt', '\t')}

	for i, (key, value) in enumerate(datasets.items()):

		path = value[0]
		file = open(path, "r")
		word_scores_human, word_scores_emb = [], []

		for line in file:
			scores = line.split()
			word_A = scores[0]
			word_B = scores[1]
			score = float(scores[2])
			if word_A in e.words and word_B in e.words:
				word_scores_human.append(((word_A, word_B), score))

		word_scores_human.sort(key=lambda x: - x[1])
		embeddings_cs = {}

		for idx, ((word_A, word_B), score) in enumerate(word_scores_human):
			cos_sim_value = cs([e.get_value(word_A)], [e.get_value(word_B)])[0][0]
			word_scores_emb.append(((word_A, word_B), cos_sim_value))
			embeddings_cs[(word_A, word_B)] = cos_sim_value

		word_scores_emb.sort(key=lambda x: - x[1])
		hum_list, emb_list = [], []

		for hum_pos, row in enumerate(word_scores_human):
			word_pair = row[0]
			hum_list.append(hum_pos)
			emb_list.append(word_scores_emb.index((word_pair, embeddings_cs[word_pair])))

		results[key] = round(spearmanr(emb_list, hum_list)[0] * 100, 2)

	return results