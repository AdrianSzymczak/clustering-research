from .config import Config
import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata

c = Config()


class Dataset:
    def consolidate_to_df(self):
        consolidated = self.results[['ID', 'title', 'snippet']]. \
            rename(columns={'ID': 'fullResultID'}). \
            set_index('fullResultID'). \
            join(self.strel.rename(
            columns={'docID': 'fullResultID', 'resultID': 'fullResultID', 'subTopicID': 'fullTopicID'}).set_index(
            'fullResultID')). \
            reset_index(). \
            assign(queryID=lambda df: df.fullResultID.apply(lambda v: v.split('.')[0]).apply(float)). \
            assign(resultID=lambda df: df.fullResultID.apply(lambda v: v.split('.')[1]).apply(float)). \
            assign(
            subtopicID=lambda df: df.fullTopicID.apply(lambda v: '0' if v is np.NaN else v.split('.')[1]).apply(float)). \
            set_index(['queryID', 'resultID']). \
            sort_index(). \
            reset_index(level=1) \
            [['title', 'snippet', 'subtopicID', 'fullResultID', 'fullTopicID', 'resultID']]. \
            join(self.topics)

        for idx in consolidated.index.unique():
            consolidated.loc[idx, 'gs_cluster'] = rankdata(consolidated.loc[idx, 'subtopicID'], method='dense') - 1

        consolidated['dataset'] = self.name
        consolidated['title'].fillna('', inplace=True)
        consolidated['snippet'].fillna('', inplace=True)
        consolidated = consolidated[consolidated.resultID <= 100.0]

        groupby_cols = ['queryID', 'dataset', 'query', 'resultID', 'title', 'snippet']

        return consolidated.reset_index()[groupby_cols + ['gs_cluster']].groupby(groupby_cols, as_index=False).agg(
            {'gs_cluster': 'min'}).sort_values(groupby_cols).set_index('queryID')


class Ambient(Dataset):
    def __init__(self, directory=c.ambient_dir):
        self.name = 'Ambient'
        self.results = pd.read_csv(os.path.join(directory, 'results.txt'), delimiter='\t', engine='python', quoting=3,
                                   converters={'ID': str})
        self.strel = pd.read_csv(os.path.join(directory, 'STRel.txt'), delimiter='\t',
                                 dtype={'subTopicID': str, 'resultID': str})
        self.subtopics = pd.read_csv(os.path.join(directory, 'subTopics.txt'), delimiter='\t', dtype={'ID': str})
        self.topics = pd.read_csv(os.path.join(directory, 'topics.txt'), delimiter='\t', dtype={'ID': float}).rename(
            columns={'ID': 'queryID', 'description': 'query'}).set_index('queryID')


class Morsque(Dataset):
    def __init__(self, directory=c.moresque_dir):
        self.name = 'Moresque'
        self.results = pd.read_csv(os.path.join(directory, 'results.txt'), delimiter='\t+', engine='python', quoting=3,
                                   converters={'ID': str})
        self.strel = pd.read_csv(os.path.join(directory, 'STRel.txt'), delimiter='\t',
                                 dtype={'subTopicID': str, 'resultID': str})
        self.subtopics = pd.read_csv(os.path.join(directory, 'subTopics.txt'), delimiter='\t', dtype={'ID': str})
        self.topics = pd.read_csv(os.path.join(directory, 'topics.txt'), delimiter='\t', dtype={'id': float}).rename(
            columns={'id': 'queryID', 'description': 'query'}).set_index('queryID')
        self.topics['query'] = self.topics['query'].apply(lambda v: ' '.join(v.split('_')))


class Odp239(Dataset):
    def __init__(self, directory=c.odp239_dir):
        self.name = 'Odp239'
        self.results = pd.read_csv(os.path.join(directory, 'docs.txt'), delimiter='\t', dtype={'ID': str})
        self.strel = pd.read_csv(os.path.join(directory, 'STRel.txt'), delimiter='\t',
                                 dtype={'subTopicID': str, 'docID': str})
        self.subtopics = pd.read_csv(os.path.join(directory, 'subTopics.txt'), delimiter='\t', dtype={'ID': str})
        self.topics = pd.read_csv(os.path.join(directory, 'topics.txt'), delimiter='\t', dtype={'ID': float}).rename(
            columns={'ID': 'queryID', 'description': 'query'}).set_index('queryID')
        self.topics['query'] = self.topics['query'].apply(lambda v: ' '.join(v.split('_')))

