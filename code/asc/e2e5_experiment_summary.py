import argparse
from asc.semantic_clustering import SimilarityProvider
from asc import SemanticClustering
import numpy as np
import pandas as pd
import logging
import time
import itertools
import marshal
import json
import os
import glob
from asc.evaluation import f_measure
from asc.evaluation import f_measure_classic
from asc.evaluation import jaccard_fast
from asc.evaluation import jaccard_custom
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  

parser = argparse.ArgumentParser(description='Neat way of storing the experiment configuration')
parser.add_argument('--results_prefix', dest='RESULTS_PREFIX', action='store',
                    help='full path to JSON that contains experiment configuration', required=True)
parser.add_argument('--summary', dest='SUMMARY', action='store',
                    help='full path to JSON that contains experiment configuration', required=True)

args = parser.parse_args()


def apply_measure(group, measure, include_others=True):

    true_labels = group.gs_cluster
    predicted_labels = group.cluster
    mask = include_others | (true_labels > 0)

    return measure(true_labels[mask], predicted_labels[mask])

summary = None

for f in glob.iglob(args.RESULTS_PREFIX + '*.pkl'):

    parts = f.split('__')
    query = parts[-1][:-4]
    dataset = parts[-2]
    logging.info("Started processing for query {} / dataset {}".format(query, dataset))

    partial_result = pd.read_pickle(f).reset_index()
    partial_result['fit_time'] = partial_result['ts_fit_transform'] - partial_result['ts_pre_fit']
    partial_result['transform_time'] = partial_result['ts_post_transform'] - partial_result['ts_fit_transform']
    partial_result['dataset'] = dataset
    partial_result['query'] = query
    partial_result.loc[partial_result['num_clusters'] == 0, 'num_clusters'] = 1
    partial_result['num_potential_clusters'] += 1

    column_list = ['queryID', 'dataset', 'param_communities', 'param_node_weight', 'param_edge_weight',
                   'param_sem_weight', 'param_other_threshold', 'param_top_k', 'fit_time', 'transform_time',
                   'num_potential_clusters', 'num_clusters']

    index_list, data_list = column_list[:8], column_list[8:]

    partial_summary_gr = partial_result.groupby(index_list)
    partial_summary = partial_summary_gr[data_list].mean()

    measures = [
        ('Jaccard', jaccard_fast),
        ('ARI', adjusted_rand_score),
        ('AMI', adjusted_mutual_info_score),
        ('F-measure', f_measure),
        ('F-measure-classic', f_measure_classic)
    ]

    starred = ['Jaccard', 'ARI', 'AMI']

    for name, func in measures:
        logging.info("Applying measure {} for query {} / dataset {}".format(name, query, dataset))
        partial_summary[name] = partial_summary_gr.apply(apply_measure, func)
        if name in starred:
            partial_summary[name + '*'] = partial_summary_gr.apply(apply_measure, func, include_others=False)

    if summary is None:
        summary = partial_summary.reset_index()
    else:
        summary = summary.append(partial_summary.reset_index(), ignore_index=True)

logging.info("Saving all results to pickle / excel file: {}".format(args.SUMMARY))

summary.to_pickle("{}.pickle".format(args.SUMMARY))

cols_to_agg = ['num_potential_clusters', 'num_clusters', 'fit_time', 'transform_time']
cols_to_agg.extend([k for k, _ in measures])
cols_to_agg.extend([k+'*' for k in starred])

summary_gr = summary.groupby(['dataset', 'param_communities', 'param_node_weight', 'param_edge_weight',
               'param_sem_weight', 'param_other_threshold', 'param_top_k'])[cols_to_agg].mean().reset_index()

summary_gr.to_excel(args.SUMMARY, sheet_name='Datasets', index=False)

logging.info("Finished")
