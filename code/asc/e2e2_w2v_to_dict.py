import argparse
from gensim.models import Word2Vec
import pandas as pd
import logging
import itertools
import marshal

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') 

parser = argparse.ArgumentParser(description='Read in gensim word2vec model, find similarities of all possible words in data and store them in custom dictionary / SimilarityProvider.')
parser.add_argument('--data', dest='DATA', action='store', help='full path to pickle that with input data', required=True)
parser.add_argument('--w2v', dest='W2V', action='store', help='full path to word2vec gensim model', required=True)
parser.add_argument('--dict', dest='DICT', action='store', help='full path to binary (marshall) that output dictionary will be stored to', required=True)
parser.add_argument('--w2v_prefix', dest='W2V_PREFIX', default='', action='store', help='prefix to add before word check, default empty string')
parser.add_argument('--chunk', dest='CHUNK', action='store',
                    help='full path to results file', required=True)
parser.add_argument('--num_chunks', dest='NUM_CHUNKS', action='store',
                    help='full path to results file', required=True)
args = parser.parse_args()

logging.info("Reading in data")
df_full = pd.read_pickle(args.DATA)

logging.info("Reading in word2vec model")
w2v = Word2Vec.load_word2vec_format(args.W2V, binary=True)

CHUNK = int(args.CHUNK)
NUM_CHUNKS = int(args.NUM_CHUNKS)


for idx in df_full.index.unique():

    if idx % NUM_CHUNKS != CHUNK:
        continue

    dictionary = dict()
    
    df = df_full.loc[idx]
    
    name = df.iat[0, 0]
    query = df.iat[0, 1]

    logging.info("Creating list of words for {} / {}".format(name, query))
    words_raw = []

    for series in [df.query_lemmatized, df.title_lemmatized, df.snippet_lemmatized]:
        for tuples_list in series:
            for word in itertools.chain(*tuples_list):
                words_raw.append(word)

    logging.info("Converting to unique ordered list of words for {} / {}".format(name, query))
    words = sorted(set(words_raw))
    len_words = len(words)
    sim_dict_size = len_words*len_words

    logging.info("Creating similarity dictionary with {} pairs for {} / {}".format(sim_dict_size, name, query))

    for i, (word1, word2) in enumerate(itertools.product(words, words)):

        if i % 500000 == 0:
            logging.info("Processed {} / {} pairs [{}, {}]".format(i, sim_dict_size, name, query))

        if word1 < word2:

            try:
                similarity = float(w2v.similarity('{}{}'.format(args.W2V_PREFIX, word1), '{}{}'.format(args.W2V_PREFIX, word2)))
                dictionary[(word1, word2)] = similarity
            except:
                pass

    logging.info("Storing dictionary object to: {}".format((args.DICT + name + query).replace(" ", "").replace(">", "")))
    with open((args.DICT + name + query).replace(" ", "").replace(">", ""), 'wb') as binary_file:
        marshal.dump(dictionary, binary_file)

logging.info("Removing word2vec model reference")
del w2v

logging.info("Finished")
