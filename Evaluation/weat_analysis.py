from sklearn.metrics.pairwise import cosine_similarity as cs
from Classes.weat import *

def average_bias_value(bias_weat_results):
	"""
	Compute bias levels for all classes.

	Parameters
	----------
	bias_weat_results: dict | Bias output, containing WEAT effect size and p value for each test within bias classes as keys.

	Returns
	-------
	means: dict | Average bias levels per each class separately.
	"""
	class_dict = {
		'gender':{('male_terms','female_terms'):[]}, 
		'race':{('black_names','white_names'):[]}, 
		'religion':{('islam_words','atheism_words'):[], ('islam_words','christianity_words'):[], ('atheism_words','christianity_words'):[]} 
	}
	
	for class_name in class_dict:
		for result in bias_weat_results[class_name]: 
			class_dict[class_name][(result[0], result[1])].append(np.abs(result[-1])) 

	means = {}    
		
	for class_name in class_dict:
		means[class_name] = round(np.mean([np.mean(class_dict[class_name][tuple_key]) for tuple_key in class_dict[class_name]]), 2)
	
	return means

def weat_analysis(embedding, bias_weat_combinations, sets, steps=-1, print_weat=False, matrices_check = True):
	"""
	For given embedding, WEAT tests combinations generate experimental WEAT results.

	Parameters
	----------
	embedding: Embedding | instance of class Embedding.
	bias_weat_combinations: dict | structure which contains subclass/target and attribute sets.
	sets: dict | structure containing all attribute and subclass/target set of words.
	steps: int | scalar representing number of iterations for generating p value (If -1, all combinations are being taken).
	print_weat: bool | Whether to print results or not.
	matrices_check: bool | Whether to compute cosine similarity values between sets.

	Returns
	-------
	final_values: dict | Effect sizes and p values for all WEAT tests
	bias_levels_d: dict | Bias levels for each class respectively
	d_values: list | List of all WEAT test effect sizes
	p_values: list | List of all WEAT test p values
	cs_matrix: dict | For each WEAT test generate mutual target-attibute sets matrix of cosine similarity values between all existing words
	"""

	final_values = {}
	p_values, d_values = [], []
	cs_matrix = {}

	#used category notation here instead of class notation (category = class)
	for category in bias_weat_combinations:

		final_values[category] = []
		d_values_category, p_values_category = [], []

		for category_target_pair in bias_weat_combinations[category]:
			for attribute_pair in bias_weat_combinations[category][category_target_pair]:
				p, d = WEAT(embedding,
							sets[category_target_pair[0]], sets[category_target_pair[1]],
							sets[attribute_pair[0]], sets[attribute_pair[1]], steps).get_stats()

				if (matrices_check==True):
					a1t1 = cs([embedding.get_value(word) for word in sets[attribute_pair[0]]], [embedding.get_value(word) for word in sets[category_target_pair[0]]])
					a2t1 = cs([embedding.get_value(word) for word in sets[attribute_pair[1]]], [embedding.get_value(word) for word in sets[category_target_pair[0]]])
					a1t2 = cs([embedding.get_value(word) for word in sets[attribute_pair[0]]], [embedding.get_value(word) for word in sets[category_target_pair[1]]])
					a2t2 = cs([embedding.get_value(word) for word in sets[attribute_pair[1]]], [embedding.get_value(word) for word in sets[category_target_pair[1]]])
					cs_matrix[(category_target_pair[0], category_target_pair[1], attribute_pair[0], attribute_pair[1])] = np.array([[a1t1, a1t2], [a2t1, a2t2]])

				if print_weat==True:
					if (np.abs(d)>0.7):
						csm = cs_matrix[(category_target_pair[0], category_target_pair[1], attribute_pair[0], attribute_pair[1])]
						cs_res = np.array([[np.mean(csm[0,0]), np.mean(csm[0,1])], [np.mean(csm[1,0]), np.mean(csm[1,1])]])
						print(f'\nBIAS: {attribute_pair[0]}, {attribute_pair[1]}, {category_target_pair[0]}, {category_target_pair[1]} : {p} ||| {"%.4f" % d} \n{cs_res}\n')
					else:
						print(f'{attribute_pair[0]}, {attribute_pair[1]}, {category_target_pair[0]}, {category_target_pair[1]} : {p} ||| {"%.4f" % d}')
				final_values[category].append([category_target_pair[0], category_target_pair[1], attribute_pair[0], attribute_pair[1], p, d])

				p_values.append(p); d_values.append(d)
				p_values_category.append(p); d_values_category.append(np.abs(d)/2)

	bias_levels_d = average_bias_value(final_values)

	return final_values, dict(sorted(bias_levels_d.items(), key=lambda x: x[1], reverse=True)), d_values, p_values, cs_matrix

def scale_bias(bias_levels):
	"""
	For centroid generation, scale bias levels must be sum up to 1.

	Parameters
	----------
	bias_levels: dict | bias levels of all classes.

	Returns
	-------
	scaled_bias_levels: dict | scaled bias levels of all classes.
	"""
	norm = np.sum(list(bias_levels.values()))
	return {key: round((value / norm), 3) for i, (key, value) in enumerate(bias_levels.items())}


def generate_centroid(scaled_bias_levels, subspace_points_dict):
	"""
	Calculate centroid

	Parameters
	----------
	scaled_bias_levels: dict | scaled bias levels of all classes.
	subspace_points_dict: dict | class definitional vectors.

	Returns
	-------
	centroid: ndarray | Centroid
	"""
	sum_of_values = []
	for subspace_weat in scaled_bias_levels:
		sum_of_values.append(scaled_bias_levels[subspace_weat] * subspace_points_dict[subspace_weat])
	return np.sum(sum_of_values, axis=0)