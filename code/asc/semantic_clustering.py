import networkx as nx
import itertools
import numpy as np

_base_graph = [None, None, None]
_current_clusters = [None, None]
_results_cache = dict()


def _custom_graph_add(g, u, v, sim, increase_weight=False):
    set_u = not g.has_node(u)
    set_v = not g.has_node(v)

    if g.has_edge(u, v) and increase_weight:
        g[u][v]['weight'] += 1
    else:
        g.add_edge(u, v, weight=0, sem_weight=sim)

    if set_u:
        g.node[u]['weight'] = 0

    if set_v:
        g.node[v]['weight'] = 0


def _filter_graph(g, node_weight_threshold=0, edge_weight_threshold=0, sem_weight_threshold=0.0):
    for u in [u for (u, d) in g.nodes(data=True) if d['weight'] < node_weight_threshold]:
        g.remove_node(u)

    for u, v in [(u, v) for (u, v, d) in g.edges(data=True) if
                 d['weight'] < edge_weight_threshold or d['sem_weight'] < sem_weight_threshold]:
        g.remove_edge(u, v)

    return g


def _update_base_graph(df, similarity_provider):
    global _base_graph
    _base_graph[2] = similarity_provider
    _base_graph[1] = df
    _base_graph[0] = nx.Graph()

    all_words = set([])
    query = set(itertools.chain(*df.loc[:, 'query_lemmatized'].iat[0]))

    for idx, row in df.iterrows():
        all_words = all_words.union(set(itertools.chain(row['title_lemmatized'], row['snippet_lemmatized'])))

    for word1list, word2list in itertools.product(all_words, all_words):

        if word1list[3] > word2list[3] and len(query.intersection(word1list)) == 0 and len(
                query.intersection(word2list)) == 0:

            max_sim = 0.0

            for word1, word2 in itertools.product(word1list, word2list):
                if word1 != word2:
                    try:
                        sim = similarity_provider.similarity(word1, word2)
                        if sim > max_sim:
                            max_sim = sim
                    except:
                        pass

            if max_sim > 0.0:
                _custom_graph_add(_base_graph[0], word1list[3], word2list[3], max_sim)

    for idx, row in df.iterrows():

        words = set(itertools.chain(row['title_lemmatized'], row['snippet_lemmatized']))

        for word1list, word2list in itertools.product(words, words):

            if word1list[3] > word2list[3] and len(query.intersection(word1list)) == 0 and len(
                    query.intersection(word2list)) == 0:

                max_sim = 0.0

                for word1, word2 in itertools.product(word1list, word2list):
                    if word1 != word2:
                        try:
                            sim = similarity_provider.similarity(word1, word2)
                            if sim > max_sim:
                                max_sim = sim
                        except:
                            pass

                if max_sim > 0.0:
                    _custom_graph_add(_base_graph[0], word1list[3], word2list[3], max_sim, increase_weight=True)

        for u in set([word[3] for word in words if word[3] not in query]):
            if _base_graph[0].has_node(u):
                _base_graph[0].node[u]['weight'] += 1


class SimilarityProvider:
    def __init__(self, dict):
        self.dict = dict

    def similarity(self, word1, word2):
        if word1 == word2:
            return 1.0
        if word1 < word2:
            return self.dict.get((word1, word2), 0.0)
        else:
            return self.dict.get((word2, word1), 0.0)


class SemanticClustering:
    def __init__(self, similarity_provider, communities, node_weight_threshold, edge_weight_threshold,
                 sem_weight_threshold, other_threshold, top_k_comparison):
        self.similarity_provider = similarity_provider
        self.communities = communities
        self.node_weight_threshold = node_weight_threshold
        self.edge_weight_threshold = edge_weight_threshold
        self.sem_weight_threshold = sem_weight_threshold
        self.other_threshold = other_threshold
        self.top_k_comparison = top_k_comparison
        self.potential_clusters = None

    def _fit(self, parameters):
        global _current_clusters

        self.graph = _filter_graph(_base_graph[0].copy(), node_weight_threshold=self.node_weight_threshold,
                                   edge_weight_threshold=self.edge_weight_threshold,
                                   sem_weight_threshold=self.sem_weight_threshold)
        if self.communities == 0:
            self.potential_clusters = [sorted(clq) for clq in nx.find_cliques(self.graph) if len(clq) >= 3]
        else:
            if _current_clusters[1][1:] == parameters[1:]:
                self.potential_clusters = [sorted(clq) for clq in
                                           nx.k_clique_communities(self.graph, 3, cliques=_current_clusters[0])]
            else:
                self.potential_clusters = [sorted(clq) for clq in nx.k_clique_communities(self.graph, 3)]
        _current_clusters[0] = self.potential_clusters
        _current_clusters[1] = parameters

    def fit(self, df):
        global _base_graph
        global _current_graph
        global _current_clusters
        parameters = (
            self.communities, self.node_weight_threshold, self.edge_weight_threshold, self.sem_weight_threshold)

        if df.equals(_base_graph[1]) and self.similarity_provider == _base_graph[2]:
            if parameters == _current_clusters[1]:
                self.potential_clusters = _current_clusters[0]
            else:
                self._fit(parameters)
        else:
            _update_base_graph(df, self.similarity_provider)
            self._fit(parameters)

    def transform(self, df):

        global _results_cache

        if self.potential_clusters is None:
            raise Exception('Fit the data before transforming')

        if self.potential_clusters == []:
            return np.zeros(len(df))

        result = []

        for idx, row in df.iterrows():

            words = set(itertools.chain(row['title_lemmatized'], row['snippet_lemmatized']))

            clusters_fits = []

            for cluster_id, potential_cluster in enumerate(self.potential_clusters):

                words_concat = ''.join(sorted(set(itertools.chain(*words))))
                potential_cluster_concat = ''.join(sorted(set(potential_cluster)))

                words_fits = _results_cache.get((words_concat, potential_cluster_concat), [])

                if not words_fits:
                    for words4 in words:
                        word_best_fit = 0.0
                        for word in words4:
                            for cword in potential_cluster:
                                similarity = 0.0
                                try:
                                    similarity = self.similarity_provider.similarity(word, cword)
                                except:
                                    pass

                                word_best_fit = np.max([word_best_fit, similarity])

                        words_fits.append(word_best_fit)

                words_fits = sorted(words_fits, reverse=True)
                _results_cache[(words_concat, potential_cluster_concat)] = words_fits
                clusters_fits.append((np.mean(words_fits[:self.top_k_comparison]), cluster_id + 1))

            clusters_fits.append((self.other_threshold, 0))
            result.append(sorted(clusters_fits, reverse=True)[0][1])

        return np.array(result)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
