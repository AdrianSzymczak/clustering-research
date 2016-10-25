import argparse
from asc.semantic_clustering import SimilarityProvider
from asc import SemanticClustering
import pandas as pd
import numpy as np
import logging
import time
import itertools
import marshal
import json
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  

parser = argparse.ArgumentParser(description='Neat way of storing the experiment configuration')
parser.add_argument('--data', dest='DATA', action='store', help='full path to pickle that with input data',
                    required=True)
parser.add_argument('--dict', dest='DICT', action='store',
                    help='full path to binary (marshall) with similarity provider', required=True)
parser.add_argument('--exp_config', dest='EXP_CONFIG', action='store',
                    help='full path to JSON that contains experiment configuration', required=True)
parser.add_argument('--results_prefix', dest='RESULTS_PREFIX', action='store',
                    help='full path to results file', required=True)
parser.add_argument('--chunk', dest='CHUNK', action='store',
                    help='full path to results file', required=True)
parser.add_argument('--num_chunks', dest='NUM_CHUNKS', action='store',
                    help='full path to results file', required=True)
args = parser.parse_args()

CHUNK = int(args.CHUNK)
NUM_CHUNKS = int(args.NUM_CHUNKS)

df_full = pd.read_pickle(args.DATA)

with open(args.EXP_CONFIG, 'r') as json_file:
    exp_config = json.load(json_file)

potential_dirs = args.RESULTS_PREFIX.split(os.pathsep)
prefix_dirs = os.path.join(*potential_dirs)
logging.info("Creating dirs: {}, if not-existend".format(prefix_dirs))
os.makedirs(prefix_dirs, exist_ok=True)

for idx in df_full.index.unique():

    if idx % NUM_CHUNKS != CHUNK:
        continue

    df = df_full.loc[idx]
    dataset = df.iat[0, 0].replace('_', '').replace(' ', '').replace('>', '')
    query = df.iat[0, 1].replace('_', '').replace(' ', '').replace('>', '')
    output_file = "{}__{}__{}.pkl".format(args.RESULTS_PREFIX, dataset, query)

    if os.path.isfile(output_file):
        logging.info("File {} already exists, skipping its creation".format(output_file))
        continue

    SemanticClustering._results_cache = dict()

    logging.info("Performing clustering for [{}]".format(idx))

    with open((args.DICT + dataset + query).replace(" ", ""), 'rb') as binary_file:
        dictionary = marshal.load(binary_file)
    sp = SimilarityProvider(dictionary)

    partial_result = None

    for node_weight, edge_weight, sem_weight, communities, other_threshold, top_k in itertools.product(
            exp_config['node_weights'], exp_config['edge_weights'],
            exp_config['sem_weights'], exp_config['communities'], exp_config['other_thresholds'],
            exp_config['top_k']):

        if communities == np.min(exp_config['communities']) and other_threshold == np.min(
                exp_config['other_thresholds']) and top_k == np.min(exp_config['top_k']):
            logging.info(
                "[{}] Performing clustering for communities / node weight / edge weight / sem_weight / other_threshold / top_k: {} / {} / {} / {} / {} / {} [{}, {}]".format(
                    idx, communities, node_weight, edge_weight, sem_weight, other_threshold, top_k, dataset, query))

        sc = SemanticClustering(sp, communities, node_weight, edge_weight, sem_weight, other_threshold, top_k)

        parameter_result = df[['dataset', 'gs_cluster']].copy()
        parameter_result['param_node_weight'] = node_weight
        parameter_result['param_edge_weight'] = edge_weight
        parameter_result['param_sem_weight'] = sem_weight
        parameter_result['param_communities'] = communities
        parameter_result['param_other_threshold'] = other_threshold
        parameter_result['param_top_k'] = top_k

        parameter_result['ts_pre_fit'] = time.time()
        sc.fit(df)
        parameter_result['ts_fit_transform'] = time.time()
        res_num_potential_clusters = len(sc.potential_clusters)
        res_num_potential_clusters = 0 if pd.isnull(res_num_potential_clusters) else res_num_potential_clusters
        parameter_result['num_potential_clusters'] = res_num_potential_clusters
        clusters = sc.transform(df)
        parameter_result['cluster'] = clusters
        res_num_clusters = len(np.unique(clusters))
        res_num_clusters = 0 if pd.isnull(res_num_clusters) else res_num_clusters
        parameter_result['num_clusters'] = res_num_clusters
        parameter_result['ts_post_transform'] = time.time()

        if partial_result is None:
            partial_result = parameter_result
        else:
            partial_result = partial_result.append(parameter_result)

    partial_result.to_pickle(output_file)
    logging.info(
        "Finished processing of query {} from dataset {}, stored to file: {}".format(dataset, query, output_file))

logging.info("Finished")
