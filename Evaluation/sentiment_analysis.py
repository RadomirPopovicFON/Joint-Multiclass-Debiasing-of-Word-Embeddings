from Utils.methods import change_the_sentences

from keras import models, layers
from sklearn import metrics
import numpy as np

class Keras_Model:

	def __init__(self, params):
		'''
		Initializing Keras Model for Sentiment Analysis Task

		Parameters
		__________
		params: dict | Information regarding number of words, max sent.length, embedding dimension
		'''
		self.model = models.Sequential()
		self.model.add(layers.Embedding(params["NB_WORDS"], params['EMB_DIM'], input_length=params['MAX_LEN']))
		self.model.add(layers.Flatten())
		self.model.add(layers.Dense(1, activation='sigmoid'))

def compare_embeddings(datasets, embeddings, params, number_of_models = 6):

	'''
	Comparing different levels of debiasing based on Sentiment Analysis task

	Parameters
	----------
	datasets: dict | From utils.get_dataset_and_dicts() method
	embeddings: dict | Keys as embedding types (original, hardweat, softweat) and Embedding instances as values
	params: dict | From sets.get_sent_analysis_sets() method
	number_of_models: int | Number of different models from which polarity score will be generated
	'''

	keys = list(datasets['targets_sets'].keys())
	emb_mod = {'No sentence modified': (datasets['x_test_padded'], datasets['y_test']),
			 'First set modification': change_the_sentences(datasets, keys[0]),
			 'Second set modification': change_the_sentences(datasets, keys[1])}
	results = {'original': [], 'hardweat': [], 'softweat': []}
	x_train_fit, y_train_fit = datasets['x_train_padded'], datasets['y_train']

	for emb_type, e in embeddings.items():

		emb_matrix = np.zeros((params['NB_WORDS'], params['EMB_DIM']))
		set_words = set(e.words)
		for word, i in datasets['word_2_index'].items():
			if word not in set_words or i >= params['NB_WORDS']: continue
			else: emb_matrix[i] = e.get_value(word);

		for j in range(0, 6):

			keras_model = Keras_Model(params)
			keras_model.model.layers[0].set_weights([emb_matrix])
			keras_model.model.layers[0].trainable = keras_model.model.layers[1].trainable = False
			keras_model.model.compile(loss = 'binary_crossentropy', optimizer='adadelta',metrics = ['accuracy']) 
			keras_model.model.fit(x_train_fit, y_train_fit, epochs=8, verbose=0)
				
			polarity_scores = {'No sentence modified':[], 'First set modification':[], 'Second set modification':[]}
			cnf_scores = {'No sentence modified':[], 'First set modification':[], 'Second set modification':[]}

			for i, (key, (x_t, y_t)) in enumerate(emb_mod.items()):

				prediction_results = keras_model.model.predict(x_t) 
				polarity_scores[key] = np.array(prediction_results)[:,0]
				prediction_results = [1 if output_instance>0.5 else 0 for output_instance in prediction_results]
				correct_results = y_t
				cnf_scores[key].append(metrics.confusion_matrix(correct_results, prediction_results))

			final_polarity = [polarity_scores['First set modification'][i]-polarity_scores['Second set modification'][i] for i in range(0, len(polarity_scores['Second set modification']))]
			results[emb_type].append((final_polarity, cnf_scores))
			print(f'F1 score for {emb_type} embedding, {j+1}.model: {round(metrics.f1_score(correct_results, prediction_results),2)}')

		print('________________________________________________________________')

	return results