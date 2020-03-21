from Evaluation.weat_analysis import *
from Utils.methods import *
from Utils.sets import *

from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.decomposition import PCA
from scipy.linalg import null_space as ns
from nltk.corpus import stopwords
from copy import deepcopy as cp
import pickle
import time

class Word_Embedding:

	def __init__(self, filename, dimension=300):
		"""
		Word Embedding initialization

		Parameters
		----------
		filename: String | directory of the .txt Embedding file
		dimension: int | Embedding dimensionality (has to correspond with .txt file)
		"""
		self.words = []
		self.vectors = []
		self.word_idx = {}
		self.dimension = dimension
		assert (filename is not None)

		word_counter = 0

		with open(filename, "r", encoding='utf8') as f:
			for i, line in enumerate(f.readlines()):
				s = line.split() 
				try:
					value = np.array([float(x) for x in s[1:]])
				except:
					continue

				word = s[0]

				if len(value)==dimension:
					self.vectors.append(value)
					self.words.append(word)
					self.word_idx[word] = word_counter
					word_counter+=1

		self.vectors = np.array(self.vectors)
		f.close()
		print(f'Regular embedding successfully read. Shape: {np.shape(self.vectors)}')

	def get_value(self, word):
		'''
		Get vector representation for particular word

		Parameters
		----------
		word: String | String representation of the word

		Returns
		-------
		vectors: list | Vector representation of the word
		'''
		try:
			return self.vectors[self.get_index_out_of_word(word)]
		except:
			print(f'Word {word} not in the Embedding.')

	def get_word_out_of_index(self, idx):
		'''
		Take word with the corresponding index from dictionary

		Parameters
		----------
		idx: int | Index of a word

		Returns
		-------
		word: String | String representation of the word
		'''
		return self.vectors[idx]

	def get_index_out_of_word(self, word):
		'''
		Take word with the corresponding index from dictionary

		Parameters
		----------
		word: String | String representation of the word

		Returns
		-------
		index: int | Index for the given word
		'''
		return self.word_idx[word]

	def normalize_vectors(self):
		'''
		Vector normalization
		'''
		self.vectors /= np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

	def get_top_k_neighbors(self, word, k):
		'''
		Find top k neighbors for a given word

		Parameters
		----------
		word: String | Word for which we ought to find closest k words
		k: int | Number of closest neighbors we ought to output

		Returns
		-------
		top_k_words : list | Top k closest words in space
		'''
		v_s = self.vectors / np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]
		query_vector = v_s[self.get_index_out_of_word(word)]
		dots = np.dot(v_s[:, np.newaxis], query_vector)
		result = {self.words[i]: round(float(dot),3) for i, dot in enumerate(dots)}
		return sorted(result.items(), key=lambda x: x[1], reverse=True)[1:(k + 1)]

	def get_center_vector(self, def_sets):
		'''
		Paper reference: https://www.aclweb.org/anthology/N19-1062/
		Code source: https://github.com/TManzini/DebiasMulticlassWordEmbedding
		'''
		means = {}
		for k, v in def_sets.items():
			wSet = []
			for w in v:
				try:
					wSet.append(self.get_value(w))
				except KeyError as e:
					pass
			set_vectors = np.array(wSet)
			means[k] = np.mean(set_vectors, axis=0)
			
		# calculate vectors to perform PCA
		matrix = []
		for k, v in def_sets.items():
			wSet = []
			for w in v:
				try:
					wSet.append(self.get_value(w))
				except KeyError as e:
					pass
			set_vectors = np.array(wSet)
			diffs = set_vectors - means[k]
			matrix.append(diffs)

		matrix = np.concatenate(matrix)

		pca = PCA(n_components=1)
		pca.fit(matrix)
		return pca.components_[0]

	def hard_weat(self, bias_levels, bias_combinations, subspace_words, sets, neighbors_threshold=1):
		'''
		HardWEAT Debiasing

		Parameters
		----------
		bias_levels: dict | Bias levels for each class respectively
		bias_combinations: dict | Classes and respective subclasses to be included in debiasing, by default in following form : {"gender" : ["male_terms", "female_terms"],   "race": ["black_names", "white_names"], "religion" : ["islam_words", "atheism_words", "christianity_words"]}
		subspace_words: set | Words within the default combination dictionary
		sets: dict | Existing attribute and target set of words
		neighbors_threshold: float | Cosine similarity float threshold for equidistancing phase
		'''
		def_vectors, subcategories_vectors, r_cat= {}, {}, 0.0000000000000000001
		temp_sets = cp(sets)
		def_sets = get_hardweat_sets()

		def_vectors = {bias_category: self.get_center_vector(def_sets[bias_category]) for bias_category in bias_combinations}
		centroid = generate_centroid(scale_bias(bias_levels), def_vectors)
		neutral_words = list(set(self.words) -subspace_words)

		start = time.time()
		print(f'Start of neutralization, there is total of {len(neutral_words)} neutral words out of total {len(self.words)} words.')
		neutral_indices = [self.word_idx[word] for word in neutral_words]
		self.vectors[neutral_indices] = neutralize_vectors(self.vectors[neutral_indices,:], centroid)
		self.normalize_vectors()
		end = time.time();
		print(f'Neutralization done in {round(end-start, 3)}s, starting with neighbor thresholding and equidistancing...')

		start = time.time()
		for key_category in def_vectors:

			subcat_keys_within_this_cat = [x for t in list(bias_combinations[key_category].keys()) for x in t]
			vectors_for_equidistancing = {key: np.zeros(self.dimension) for key in subcat_keys_within_this_cat}
			center_vector = neutralize_vectors(def_vectors[key_category], centroid)
			equidistant_def_subcat_vectors_dict = make_vectors_equidistant(center_vector, vectors_for_equidistancing, r_cat)

			for i, key_subcategory in enumerate(subcat_keys_within_this_cat):
				values_okay, values_not_ok_idx, values_not_ok_idx_max = False, 0, 500
				while not values_okay:

					r_subcat = random.randint(1, 2**16-1)
					new_vectors = make_vectors_equidistant(equidistant_def_subcat_vectors_dict[key_subcategory], {word: self.get_value(word) for word in temp_sets[key_subcategory]}, r_subcat)

					found_artifact = False
					matrix_of_similarities = cs(np.float16(list(new_vectors.values())), np.float16(self.vectors))

					for ravel_idx, cs_value in enumerate(np.ravel(matrix_of_similarities)):
						if (cs_value>neighbors_threshold):
							found_artifact = True
							values_not_ok_idx+=1
							if(values_not_ok_idx % 100==0):
								print(f'{values_not_ok_idx} unsuccessful equdistancing iterations for {key_subcategory}')
							break

					if (found_artifact == True and values_not_ok_idx<values_not_ok_idx_max):
						values_okay = False
					else:
						for key in new_vectors:
							self.vectors[self.get_index_out_of_word(key)] = new_vectors[key]
						values_okay = True
						if (values_not_ok_idx>=values_not_ok_idx_max):
							print(f'Could not perform equidistancing below the requested threshold for {key_subcategory}')

			print(f'Finished with all {key_category} subcategories')

		self.normalize_vectors()
		self.vectors = np.array(self.vectors)
		end = time.time();
		print(f'Equidistancing done in {round(end-start, 3)}s.')

	def soft_weat(self, sets, target_at_dict, bias_combinations, l=1, nullspace_iterations = -1, neighb_count=20):
		'''
		SoftWEAT Debiasing

		Parameters
		----------
		sets: Existing attribute and target set of words
		target_at_dict: dict | Keys being target/subclass sets and values corresponding attribute sets for debiasing
		bias_combinations: dict | Classes and respective subclasses to be included in debiasing, by default in following form : {"gender" : ["male_terms", "female_terms"],   "race": ["black_names", "white_names"], "religion" : ["islam_words", "atheism_words", "christianity_words"]}
		l: float | Trade-off parameter (1 - Highest level of removal, 0 - lowest)
		nullspace_iterations: int | Number of nullspaces to be included in iterative bias minimization. If -1, all are taken into consideration
		neighb_count: int | Number of neighbors that initial target/subclass lists are expanded on
		'''

		nullspace_dict, neighbors, subclasses, duplicates = {}, {}, [], {} 
		target_words_complete = list(dict.fromkeys([word for class_name in target_at_dict for word in sets[class_name]]))
		substopwords = set(stopwords.words('english')) - set(target_words_complete)
		attribute_sets_complete = set([word for subclass_name in target_at_dict for a_set in target_at_dict[subclass_name]for word in sets[a_set]])

		cs_matrix = cs([self.get_value(word) for word in target_words_complete], self.vectors)
		cs_idx = {word:idx for idx, word in enumerate(target_words_complete)}
		dictionary_categories = {"gender" : ["male_terms", "female_terms"],   "race": ["black_names", "white_names"], "religion" : ["islam_words", "atheism_words", "christianity_words"]}

		#Generating neighbors
		for class_name in dictionary_categories.keys():
			for subclass_name in dictionary_categories[class_name]:

				if subclass_name not in target_at_dict: continue
				
				subclasses.append(subclass_name)
				neighbors[subclass_name] = set(sets[subclass_name])
				for word in sets[subclass_name]:
					new_words = set([self.words[neighbor_idx] for neighbor_idx in (cs_matrix[cs_idx[word],:].argsort()[-neighb_count:]) if cs([self.vectors[neighbor_idx]], [self.get_value(word)]) > 0.6])
					new_words -= set([word for class_name, subclasses in dictionary_categories.items() for subcl in subclasses for word in sets[subcl] if subcl!=subclass_name])
					neighbors[subclass_name] = neighbors[subclass_name].union(new_words)

					#Identifying duplicates
					for word in new_words:
						for subclass in subclasses:
							if word in neighbors[subclass] and subclass!=subclass_name:
								if word not in duplicates:
									duplicates[word] = set()
								duplicates[word].add(subclass)
								duplicates[word].add(subclass_name)
		
		#Removing duplicates
		for word_dup in duplicates:
			word_dup_subclasses = list(duplicates[word_dup])
			cs_subclasses = [cs([np.sum([self.get_value(w) for w in sets[subclass_name]], axis = 0)], [self.get_value(word_dup)]) for subclass_name in word_dup_subclasses]
			for sc in [x for i,x in enumerate(word_dup_subclasses) if i!=np.argmax(cs_subclasses)]:
				neighbors[sc].remove(word_dup)

		for class_name in dictionary_categories.keys():

			#Iterating through target set that might contain bias towards some attribute sets of words 
			for subclass_name in dictionary_categories[class_name]:

				if subclass_name not in target_at_dict.keys(): 
					continue 

				attribute_set_names = list(target_at_dict[subclass_name]); attribute_set_names.sort()
				all_words_for_subclass = list( neighbors[subclass_name] - attribute_sets_complete - substopwords)
				mean_value = np.mean([self.get_value(word) for word in all_words_for_subclass], axis=0)
				vectors_for_nullspacing = np.array([self.get_wordset_mean(sets[a_set]) for a_set in attribute_set_names])
				null_space = ns(vectors_for_nullspacing)

				bias_levels_per_nullspace = []
				#print(f'Words for subclass {subclass_name} len: {(all_words_for_subclass)}')

				no_of_iterations = np.size(null_space, 1) if nullspace_iterations==-1 else nullspace_iterations
				for k in range(0, no_of_iterations):

					e = cp(self)
					nullspace_dict[subclass_name] = null_space[:,k] 
					T = nullspace_dict[subclass_name] -  mean_value
					T = make_translation_matrix(T, l)
					vectors_for_translation = np.vstack([np.transpose([e.get_value(word) for word in all_words_for_subclass]), np.ones((1, len(all_words_for_subclass)))])
					transformed_points = np.matmul(T, vectors_for_translation)
					for i, word in enumerate(all_words_for_subclass):  
						e.vectors[e.get_index_out_of_word(word)] = transformed_points[0:-1,i]
					_, bias_levels_d, _, _, _ = weat_analysis(e, bias_combinations, sets, steps=1000)
					bias_levels_per_nullspace.append(bias_levels_d[class_name])
					del e

				min_nullspace_key = np.argmin(bias_levels_per_nullspace)
				final_t_vector = null_space[:,min_nullspace_key] - mean_value
				T_final = make_translation_matrix(final_t_vector, l)
				vectors_for_translation = np.vstack([np.transpose([self.get_value(word) for word in all_words_for_subclass]), np.ones((1, len(all_words_for_subclass)))])
				transformed_points = np.matmul(T_final, vectors_for_translation)

				for i, word in enumerate(all_words_for_subclass):  
					self.vectors[self.get_index_out_of_word(word)] = transformed_points[0:-1,i]
				
				print(f'Subclass {subclass_name} finished.')
				
		self.normalize_vectors()

	def get_wordset_mean(self, set_of_words):
		"""
		For Embedding and given set of words, find average vector

		Parameters
		----------
		embedding: Embedding | Word Embedding instance
		set_of_words: list | words which will be averaged based on their vector position

		Returns
		-------
		set_mean: ndarray | Average value of all given word representations
		"""
		matrix = [self.get_value(word) for word in set_of_words]
		return np.mean(matrix, axis=0)

	def reduce_dim_version_of_embeddings(self, dimension=3):
		'''
		Reduce dimensionality of embeddings via PCA

		Parameters
		----------
		dimension: dimensionality to which embeddings will be reduced
		'''
		self.vectors = PCA(n_components=dimension).fit_transform(self.vectors)

	def save_embedding(self, filename, pkl_format=False):
		'''
		Save Embedding in pkl/txt format

		Parameters
		----------
		filename: String | directory, filename to which output will be generated
		pkl_format: bool | Determining whether embeddings will be saves in .txt or .pkl format
		'''
		if(pkl_format==True):
			print(f'entered pkl method for file: {filename}')
			output = open(f'{filename}.pkl', 'wb')
			pickle.dump(self, output)
			output.close()
		else:
			print(f'entered txt method for file: {filename}')
			with open(filename, 'w') as file:
				for i in range(0, len(self.vectors)):
					vector_string = f'{self.words[i]} ' + ' '.join([str(np.float16(x)) for x in self.vectors[i]])
					if (i != len(self.vectors)-1):
						file.write(f'{vector_string}\n')
					else:
						file.write(f'{vector_string}')
				print('Done.')
			file.close()