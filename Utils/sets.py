import itertools

def get_def_seeds(sets_of_existing_words):
    '''
    Receive definitional sets for HardWEAT purposes

    Parameters
    ----------
    sets_of_existing_words: dict | Key representing a subclass/target set, and value set of words

    Returns
    -------
    extended_sets: set | All non-neutral words
    '''

    #Source: https://github.com/uclanlp/gn_glove
    male_gn_glove = "countryman fraternal wizards manservant fathers divo actor bachelor papa dukes barman countrymen brideprice hosts airmen andropause penis prince governors abbot men widower gentlemen sorcerers sir bridegrooms baron househusbands gods nephew widowers lord brother grooms priest adultors andrology bellboys his marquis princes emperors stallion chairman monastery priests boyhood fellas king dudes daddies manservant semen spokesman tailor cowboys dude bachelors barbershop emperor daddy masculism guys enchanter guy fatherhood androgen cameramen godfather strongman god patriarch uncle chairmen sir brotherhood host testosterone husband dad steward males cialis spokesmen pa beau stud bachelor wizard sir nephews fathered bull beaus councilmen landlords grandson fiances stepfathers horsemen grandfathers adultor schoolboy rooster grandsons bachelor cameraman dads him master lad policeman monk actors salesmen boyfriend councilman fella statesman paternal chap landlord brethren lords blokes fraternity bellboy duke ballet_dancer dudes fiance colts husbands suitor paternity he businessman masseurs hero deer busboys boyfriends kings brothers masters stepfather grooms son studs cowboy mentleman sons baritone salesman paramour male_host monks menservants mr. headmasters lads congressman airman househusband priest barmen barons abbots handyman beard fraternities stewards colt czar stepsons himself boys lions gentleman penis his masseur bulls uncles bloke beards hubby lion sorcerer macho father gays male waiters sperm prostate stepson prostatic_utricle businessmen heir waiter headmaster man governor god bridegroom grandpa groom dude gay gents boy grandfather gelding paternity roosters prostatic_utricle priests manservants stailor busboy heros".split(" ")
    female_gn_glove = "countrywoman sororal witches maidservant mothers diva actress spinster mama duchesses barwoman countrywomen dowry hostesses airwomen menopause clitoris princess governesses abbess women widow ladies sorceresses madam brides baroness housewives godesses niece widows lady sister brides nun adultresses obstetrics bellgirls her marchioness princesses empresses mare chairwoman convent priestesses girlhood ladies queen gals mommies maid female_ejaculation spokeswoman seamstress cowgirls chick spinsters hair_salon empress mommy feminism gals enchantress gal motherhood estrogen camerawomen godmother strongwoman goddess matriarch aunt chairwomen ma'am sisterhood hostess estradiol wife mom stewardess females viagra spokeswomen ma belle minx maiden witch miss nieces mothered cow belles councilwomen landladies granddaughter fiancees stepmothers horsewomen grandmothers adultress schoolgirl hen granddaughters bachelorette camerawoman moms her mistress lass policewoman nun actresses saleswomen girlfriend councilwoman lady stateswoman maternal lass landlady sistren ladies wenches sorority bellgirl duchess ballerina chicks fiancee fillies wives suitress maternity she businesswoman masseuses heroine doe busgirls girlfriends queens sisters mistresses stepmother brides daughter minxes cowgirl lady daughters mezzo saleswoman mistress hostess nuns maids mrs. headmistresses lasses congresswoman airwoman housewife priestess barwomen barnoesses abbesses handywoman toque sororities stewardesses filly czarina stepdaughters herself girls lionesses lady vagina hers masseuse cows aunts wench toques wife lioness sorceress effeminate mother lesbians female waitresses ovum skene_gland stepdaughter womb businesswomen heiress waitress headmistress woman governess godess bride grandma bride gal lesbian ladies girl grandmother mare maternity hens uterus nuns maidservants seamstress' busgirl heroines".split(" ")

    #Source: https://github.com/uclanlp/corefBias
    wino = "cowboy cowgirl cowboys cowgirls camerawomen cameramen cameraman camerawoman busboy busgirl busboys busgirls bellboy bellgirl bellboys bellgirls barman barwoman barmen barwomen tailor seamstress tailors seamstress' prince princess princes princesses governor governess governors governesses adultor adultress adultors adultresses god godess gods godesses host hostess hosts hostesses abbot abbess abbots abbesses actor actress actors actresses bachelor spinster bachelors spinsters baron baroness barons barnoesses beau belle beaus belles bridegroom bride bridegrooms brides brother sister brothers sisters duke duchess dukes duchesses emperor empress emperors empresses enchanter enchantress father mother fathers mothers fiance fiancee fiances fiancees priest nun priests nuns gentleman lady gentlemen ladies grandfather grandmother grandfathers grandmothers headmaster headmistress headmasters headmistresses hero heroine heros heroines lad lass lads lasses landlord landlady landlords landladies male female males females man woman men women manservant maidservant manservants maidservants marquis marchioness masseur masseuse masseurs masseuses master mistress masters mistresses monk nun monks nuns nephew niece nephews nieces priest priestess priests priestesses sorcerer sorceress sorcerers sorceresses stepfather stepmother stepfathers stepmothers stepson stepdaughter stepsons stepdaughters steward stewardess stewards stewardesses uncle aunt uncles aunts waiter waitress waiters waitresses widower widow widowers widows wizard witch wizards witches".split(" ")

    seed_sets = {}
    #Source: https://github.com/TManzini/DebiasMulticlassWordEmbeddings
    seed_sets['religion'] = [ "synagogue", "synagogues", "altar", "altars", "parish", "parishes", "biblical", "bishop", "bishops", "jihadist", "clergy", "bible", "bibles", "mosque", "mosques", "mullah", "church", "churches", "sermon", "sermons", "papacy", "imam", "pew", "chancel", "pope", "priest", "priests", "baptism", "jihad", "confessional", "holy_eucharist", "evangelical", "jesus", "burqa", "vicar", "vicars", "judaism", "christianity", "islam", "jew", "christian", "muslim", "torah", "quran", "rabbi" ]
    seed_sets['race'] = ["black", "african", "africa"]
    seed_sets['gender'] = []

    for set_name in ['atheism_words', 'islam_words', 'christianity_words']:
        for word in sets_of_existing_words[set_name]:
            seed_sets['religion'].append(word)

    for set_name in ['black_names', 'white_names']:
        for word in sets_of_existing_words[set_name]:
            seed_sets['race'].append(word)

    for set_name in ['male_terms', 'female_terms']:
        for word in sets_of_existing_words[set_name]:
            seed_sets['gender'].append(word)

    for set_x in [male_gn_glove, female_gn_glove, wino]:
        for word in set_x:
            seed_sets['gender'].append(word)

    return set([word for set_name in seed_sets for word in seed_sets[set_name]])


def get_hardweat_sets():

    '''
    Paper reference: https://www.aclweb.org/anthology/N19-1062/
    Code source: https://github.com/TManzini/DebiasMulticlassWordEmbedding
    (*Instead of judaism, atheism has been used, could be modified diferently*)
    '''

    def_sets = {}

    def_sets['gender'] = {i: v for i, v in enumerate([["he", "she"],
    ["his", "hers"],
    ["son", "daughter"],
    ["father", "mother"],
    ["male", "female"],
    ["boy", "girl"],
    ["uncle", "aunt"]])}

    def_sets['race'] = {i: v for i, v in enumerate([["black", "caucasian"],
    ["african", "caucasian"],
    ["black", "white"],
    ["africa", "america"],
    ["africa", "europe"]])}

    def_sets['religion'] = {i: v for i, v in enumerate([
    ["atheism", "christianity", "islam"],
    ["atheist", "christian", "muslim"],
    ["university", "church", "mosque"],
    ["scientist", "priest", "imam"]])}

    return def_sets

def get_sets(all_words=None, target_set_reduction=True):

    '''
    Receive attribute and target sets for HardWEAT

    Parameters
    ----------
    all_words: set | All existing words from embedding (Prevent from accessing non-existing words)
    target_set_reduction: bool | if True, we limit the length of target sets to be equal to the length of the shortest target set

    Returns
    -------
    attribute_sets: dict | Dictionary of attribute sets
    attribute_sets_pairs: dict | Corresponding opposite attribute sets taken as keys and words as values
    targets_sets: dict | Dictionary of subclass/target sets
    '''

    dictionary_categories = {"gender" : ["male_terms", "female_terms"], "race": ["black_names", "white_names"], "religion" : ["islam_words", "atheism_words", "christianity_words"]}
    present_target_sets = list({x for v in dictionary_categories.values() for x in v})
    attribute_sets = {}
    targets_sets = {}
    attribute_sets_pairs = {}

    # Sources: http://science.sciencemag.org/content/356/6334/183 , https://arxiv.org/pdf/1608.07187.pdf
    attribute_sets['pleasant'] = ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal',
                        'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor',
                        'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation', 'joy', 'wonderful']
    attribute_sets['unpleasant'] = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison',
                          'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty',
                          'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison', 'terrible', 'horrible',
                          'nasty', 'evil', 'war', 'awful', 'failure']
    attribute_sets['instruments'] = ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica',
                           'mandolin', 'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle',
                           'harpsichord', 'piano', 'viola', 'bongo', 'flute', 'horn', 'saxophone', 'violin']
    attribute_sets['weapons'] = ['arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword',
                       'blade', 'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun',
                       'teargas', 'cannon', 'grenade', 'mace', 'slingshot', 'whip']
    targets_sets['white_names'] = ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin', 'ryan', 'andrew',
                           'fred', 'jack', 'matthew', 'stephen', 'brad', 'jed', 'todd', 'brandon',
                           'hank', 'jonathan', 'peter', 'wilbur', 'amanda', 'courtney', 'heather', 'melanie', 'sara',
                           'amber', 'crystal', 'katie', 'meredith', 'shannon', 'betsy', 'kristin', 'nancy',
                           'stephanie', 'bobbie-sue', 'ellen', 'lauren', 'peggy', 'sue-ellen', 'colleen', 'emily',
                           'megan', 'rachel', 'wendy', 'brendan', 'geoffrey', 'brett', 'jay', 'neil', 'anne', 'carrie',
                           'jill', 'laurie', 'kristen', 'paul', 'greg', 'sarah', 'donna']
    targets_sets['black_names'] = ['alonzo', 'jamel', 'lerone', 'percell', 'theo', 'alphonse', 'jerome', 'leroy', 'rasaan',
                           'torrance', 'darnell', 'lamar', 'lionel', 'rashaun', 'tyree', 'lamont', 'malik',
                           'terrence', 'tyrone', 'everol', 'marcellus', 'terryl', 'wardell', 'aiesha',
                           'lashelle', 'nichelle', 'shereen', 'temeka', 'ebony', 'latisha', 'shaniqua', 'tameisha',
                           'teretha', 'jasmine', 'latonya', 'shanise', 'tanisha', 'tia', 'lakisha', 'latoya',
                           'tashika', 'yolanda', 'lashandra', 'malika', 'tawanda', 'yvette', 'hakim',
                           'jermaine', 'kareem', 'jamal', 'rasheed', 'aisha', 'keisha', 'kenya', 'tamika','shavonn',  'sharise', 'deion',  'lavon']

    attribute_sets['career'] = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'] 
    attribute_sets['family'] = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
    attribute_sets['science'] = ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy']
    attribute_sets['art'] = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture', 'shakespeare']
    targets_sets['male_terms'] = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son', 'father', 'uncle', 'grandfather']
    targets_sets['female_terms'] = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'mother', 'aunt', 'grandmother']

    # Source: https://arxiv.org/pdf/1711.08412.pdf
    attribute_sets['intellectual_words'] = ['resourceful', 'inquisitive', 'sagacious', 'inventive', 
                                  'adaptable', 'intuitive',
                                  'analytical', 'imaginative', 'shrewd', 'thoughtful',
                                  'smart', 'clever', 'brilliant', 'logical', 'intelligent', 'apt',
                                  'genius', 'wise', 'stupid', 'dumb', 'dull', 'clumsy', 'foolish', 'naive', 'unintelligent']

    attribute_sets['appearance_words'] =  ['alluring', 'voluptuous', 'blushing', 'homely', 'plump', 'sensual', 'gorgeous', 'slim',
                                'bald', 'athletic', 'fashionable', 'stout', 'ugly', 'muscular', 'slender', 'feeble',
                                'handsome', 'healthy', 'attractive', 'fat', 'weak', 'thin', 'pretty', 'beautiful',
                                'strong']
    targets_sets['islam_words'] = ['allah', 'ramadan', 'turban', 'emir', 'salaam', 'sunni', 'koran', 'imam', 'sultan', 'prophet', 'veil', 'ayatollah', 'shiite', 'mosque', 'islam', 'sheik', 'muslim', 'muhammad']
    targets_sets['christianity_words'] = ['baptism', 'messiah', 'catholicism', 'resurrection', 'christianity', 'salvation', 'protestant', 'gospel', 'trinity', 'jesus', 'christ', 'christian', 'cross','catholic', 'church']
    targets_sets['atheism_words'] = ['atheism', 'atheist', 'atheistic', 'heliocentric', 'evolution', 'darwin', 'galilei', 'agnostic', 'agnosticism', 'pagan', 'infidel', 'disbelief', 'scepticism', 'philosophy','university', 'kopernikus']

    # Source: https://arxiv.org/pdf/1903.10561.pdf
    attribute_sets['shy'] = ["soft","quiet","compromising","rational","calm", "kind", "agreeable", "servile", "pleasant", "cautious", "friendly", "supportive","nice","mild","demure","passive","indifferent", "submissive"]
    attribute_sets['aggressive'] = ["shrill","loud","argumentative","irrational","angry","abusive","obnoxious","controlling","nagging","brash","hostile","emasculating","mean","harsh","sassy","aggressive","opinionated","domineering"]
    attribute_sets['competent'] = ["competent","productive","effective","ambitious","active","decisive","strong","tough","bold","assertive"]
    attribute_sets['incompetent'] = ["incompetent", "unproductive", "ineffective","unambitious","passive","indecisive","weak","gentle", "timid", "unassertive"]
    attribute_sets['likeable'] = ["agreeable","fair","honest","trustworthy","selfless","accommodating","likable","liked"]
    attribute_sets['unlikeable'] = ["abrasive","conniving","manipulative","dishonest","selfish","pushy","unlikable","unliked"]

    if(all_words==None):
        for key in attribute_sets:
            attribute_sets[key] = [word.lower() for word in attribute_sets[key]]
    else:
        for key in attribute_sets:
            attribute_sets[key] = [word.lower() for word in attribute_sets[key] if word in all_words]

    attribute_sets_pairs[('likeable', 'unlikeable')] = (attribute_sets['likeable'], attribute_sets['unlikeable'])
    attribute_sets_pairs[('competent', 'incompetent')] = (attribute_sets['competent'], attribute_sets['incompetent'])
    attribute_sets_pairs[('shy', 'aggressive')] = (attribute_sets['shy'], attribute_sets['aggressive'])
    attribute_sets_pairs[('intellectual_words', 'appearance_words')] = (attribute_sets['intellectual_words'], attribute_sets['appearance_words'])
    attribute_sets_pairs[('family', 'career')] = (attribute_sets['family'], attribute_sets['career'])
    attribute_sets_pairs[('instruments', 'weapons')] = (attribute_sets['instruments'], attribute_sets['weapons'])
    attribute_sets_pairs[('pleasant', 'unpleasant')] = (attribute_sets['pleasant'], attribute_sets['unpleasant'])
    attribute_sets_pairs[('science', 'art')] = (attribute_sets['science'], attribute_sets['art'])

    #Equalizing number of elements within attribute sets
    for pair_key, pair_value in attribute_sets_pairs.items():
        min_len = min(len(pair_value[0]), len(pair_value[1]))
        attribute_sets[pair_key[0]] = attribute_sets[pair_key[0]][0:min_len]
        attribute_sets[pair_key[1]] = attribute_sets[pair_key[1]][0:min_len]

    if(all_words==None):
        for key in targets_sets:
            targets_sets[key] = [word.lower() for word in targets_sets[key]]
    else:
        for key in targets_sets:
            targets_sets[key] = [word.lower() for word in targets_sets[key] if word in all_words]

    if (target_set_reduction == True):
        for class_name in dictionary_categories:
            class_targets = [targets_sets[s] for s in dictionary_categories[class_name]]
            min_length = min([len(t) for t in class_targets])
            target_set_len_dictionary = {s:len(targets_sets[s]) for s in present_target_sets}
            if(min_length < 5):
                print(f'{target_set_len_dictionary} => Minimum length is {min_length} for {class_name}. Consider not using some of the target sets.')
            for set_name in dictionary_categories[class_name]:
                targets_sets[set_name] = targets_sets[set_name][:min_length]

    return attribute_sets, attribute_sets_pairs, targets_sets

def get_sent_analysis_sets():

    """
    Generate structures for Sentiment Analysis task

    Returns
    -------
    dictionary_categories: dict | Bias class as keys and respective subclasses as values
    parameters_dict: dict | Input model information
    target_sets_dict: dict | Dictionary containing existing words for each Word Embedding model within a subclass pair
    """

    dictionary_categories = {"gender": ["male_terms", "female_terms"], "race": ["black_names", "white_names"],
                             "religion": ["islam_words", "atheism_words", "christianity_words"]}

    parameters_dict = \
        {
            "NB_WORDS": 124252,
            "MAX_LEN": 50,
            "NO_PRETR_DIM": 100,
            "EMB_DIM": 300
        }

    target_sets_dict = {
        'black_white':
            {
                'word2vec':
                    {
                        'first_set': ['theo', 'jerome', 'leroy', 'lamar', 'lionel', 'malik', 'ebony', 'jasmine', 'tia',
                                      'hakim', 'kareem', 'jamal', 'kenya'],
                        'second_set': ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin',
                                       'ryan', 'andrew', 'fred', 'jack']
                    },
                'fasttext':
                    {
                        'first_set': ['alonzo', 'jamel', 'theo', 'alphonse', 'jerome', 'leroy', 'torrance', 'darnell',
                                      'lamar', 'lionel', 'tyree', 'lamont', 'malik', 'terrence', 'tyrone', 'marcellus',
                                      'ebony', 'jasmine', 'tanisha', 'tia', 'latoya', 'yolanda', 'malika', 'yvette',
                                      'hakim', 'jermaine', 'kareem', 'jamal', 'aisha', 'keisha', 'kenya'],
                        'second_set': ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin',
                                       'ryan', 'andrew', 'fred', 'jack', 'matthew', 'stephen', 'brad', 'jed', 'todd',
                                       'brandon', 'hank', 'jonathan', 'peter', 'wilbur', 'amanda', 'courtney',
                                       'heather', 'melanie', 'sara', 'amber', 'crystal', 'katie', 'meredith', 'shannon',
                                       'betsy']
                    },
                'glove':
                    {
                        'first_set': ['alonzo', 'jamel', 'theo', 'alphonse', 'jerome', 'leroy', 'torrance', 'darnell',
                                      'lamar', 'lionel', 'tyree', 'lamont', 'malik', 'terrence', 'tyrone', 'marcellus',
                                      'ebony', 'jasmine', 'tanisha', 'tia', 'latoya', 'yolanda', 'malika', 'yvette',
                                      'hakim', 'jermaine', 'kareem', 'jamal', 'aisha', 'keisha', 'kenya'],
                        'second_set': ['adam', 'chip', 'harry', 'josh', 'roger', 'alan', 'frank', 'ian', 'justin',
                                       'ryan', 'andrew', 'fred', 'jack', 'matthew', 'stephen', 'brad', 'jed', 'todd',
                                       'brandon', 'hank', 'jonathan', 'peter', 'wilbur', 'amanda', 'courtney',
                                       'heather', 'melanie', 'sara', 'amber', 'crystal', 'katie', 'meredith', 'shannon',
                                       'betsy', 'kristin', 'nancy', 'stephanie', 'ellen', 'lauren', 'peggy', 'colleen']
                    }
            },
        'male_female':
            {
                'glove':
                    {'first_set': ['male', 'man', 'boy', 'brother', 'him', 'son', 'father', 'uncle', 'grandfather'],
                     'second_set': ['female', 'woman', 'girl', 'sister', 'her', 'daughter', 'mother', 'aunt',
                                    'grandmother']},
                'word2vec':
                    {'first_set': ['male', 'man', 'boy', 'brother', 'him', 'son', 'father', 'uncle', 'grandfather'],
                     'second_set': ['female', 'woman', 'girl', 'sister', 'her', 'daughter', 'mother', 'aunt',
                                    'grandmother']},
                'fasttext':
                    {'first_set': ['male', 'man', 'boy', 'brother', 'him', 'son', 'father', 'uncle', 'grandfather'],
                     'second_set': ['female', 'woman', 'girl', 'sister', 'her', 'daughter', 'mother', 'aunt',
                                    'grandmother']}
            },
        'islam_christianity':
            {
                "fasttext":
                    {"first_set": ["allah", "ramadan", "emir", "salaam", "koran", "imam", "sultan", "prophet", "veil",
                                   "ayatollah", "mosque"],
                     "second_set": ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation",
                                    "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross",
                                    "catholic"]
                     },
                "glove":
                    {"first_set": ["allah", "ramadan", "emir", "salaam", "koran", "imam", "sultan", "prophet", "veil",
                                   "ayatollah", "mosque", "islam"],
                     "second_set": ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation",
                                    "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross",
                                    "catholic", "church"]
                     },
                "word2vec": {
                    "first_set": ["allah", "ramadan", "emir", "salaam", "koran", "imam", "sultan", "prophet", "veil",
                                  "ayatollah"],
                    "second_set": ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation",
                                   "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross"]}
            }
    }

    return dictionary_categories, parameters_dict, target_sets_dict

def get_combinations(sets, attribute_sets_pairs):
    '''
    Generate structure format necessary for WEAT analysis (WEAT_processing.weat_analysis method)

    Parameters
    ----------
    dictionary_categories_original: dict | Bias classes as keys and subclasses as values
    sets: dict | All subclass/target sets of words
    attribute_sets_pairs: dict | Tuples of attribute sets used within WEAT experiments

    Returns
    -------
    dictionary_categories: dict | Initial structure for WEAT analysis
    '''

    dictionary_categories = {"gender" : ["male_terms", "female_terms"],   "race": ["black_names", "white_names"], "religion" : ["islam_words", "atheism_words", "christianity_words"]}
    subcategories_to_be_deleted = []
    categories_to_be_deleted = []

    attribute_dict = {}
    for pair in attribute_sets_pairs:
        if (len(sets[pair[0]]) != 0 and len(sets[pair[0]]) != 0):
            attribute_dict[(pair)] = 0.5

    combination_idx = 0
    for key in dictionary_categories:
        for i in range(0, len(dictionary_categories[key])):
            if ((dictionary_categories[key][i] not in sets) or len(sets[dictionary_categories[key][i]]) == 0):
                subcategories_to_be_deleted.append(dictionary_categories[key][i])
        [dictionary_categories[key].remove(item_for_deletion) for item_for_deletion in subcategories_to_be_deleted
         if item_for_deletion in dictionary_categories[key]] #Item for deletion may belong to different category

        combinations = []
        for comb in itertools.combinations(range(0, len(dictionary_categories[key])), 2):
            left, right = comb
            combinations.append((dictionary_categories[key][left], dictionary_categories[key][right]))

        dictionary_categories[key] = {combination: attribute_dict for combination in combinations}

        if (len(list(dictionary_categories[key].keys())) == 0):
            categories_to_be_deleted.append(key)
    [dictionary_categories.pop(key) for key in categories_to_be_deleted]

    return dictionary_categories