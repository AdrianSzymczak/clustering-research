import argparse
import asc
import asc.text_processing as tp
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') 

parser = argparse.ArgumentParser(description='Read the Ambient, Moresque and Odp239, append them and store in pickle.')
parser.add_argument('--data', dest='DATA', action='store', help='full path to pickle that output data will be stored to', required=True)
args = parser.parse_args()

logging.info("Reading in Ambient dataset")
ambient = asc.Ambient()
logging.info("Reading in Moresque dataset")
morsque = asc.Morsque()
logging.info("Reading in Odp239 dataset")
odp239 = asc.Odp239()

logging.info("Consolidating Ambient dataset")
ambient_df = ambient.consolidate_to_df()
logging.info("Consolidating Moresque dataset")
morsque_df = morsque.consolidate_to_df()
logging.info("Consolidating Odp239 dataset")
odp239_df = odp239.consolidate_to_df()

odp239_df.index += morsque_df.index.max()

logging.info("Concatenating datasets into single Data Frame")
df = ambient_df.append(morsque_df).append(odp239_df)

logging.info("Processing query")
df['query_lemmatized'] = df['query']. \
    apply(tp.tokenize). \
    apply(tp.part_of_speech). \
    apply(tp.filter_irrelevant_tokens). \
    apply(tp.lemmatize)

logging.info("Processing title")
df['title_lemmatized'] = df.title. \
    apply(tp.tokenize). \
    apply(tp.part_of_speech). \
    apply(tp.filter_irrelevant_tokens). \
    apply(tp.lemmatize)

logging.info("Processing snippet")
df['snippet_lemmatized'] = df.snippet. \
    apply(tp.tokenize). \
    apply(tp.part_of_speech). \
    apply(tp.filter_irrelevant_tokens). \
    apply(tp.lemmatize)

df = df[['dataset', 'query', 'title', 'snippet', 'query_lemmatized', 'title_lemmatized', 'snippet_lemmatized', 'gs_cluster']]
logging.info("Saving processed Data Frame to file: {}".format(args.DATA))
df.to_pickle(args.DATA)
logging.info("Finished")
