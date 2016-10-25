import itertools
import numpy as np
import pandas as pd



def f_measure(labels_true, labels_pred):

    if np.sum(labels_true != 0) == 0:
        return 0.0

    init_df = pd.DataFrame({'true': labels_true, 'cluster': labels_pred})

    n = len(init_df)
    n0 = len(init_df.query('true == 0'))

    def bcnt(s):
        bct = np.bincount(np.int64(s))
        max_all = np.max(bct)
        max_without_zero = 0 if len(bct) == 1 else np.max(bct[1:])

        if max_without_zero >= max_all:
            return max_without_zero
        else:
            return max_without_zero

    df = init_df.groupby('cluster').agg({'true': bcnt})

    precision = df.true.sum() / n
    recall = df.true.sum() / (n - n0)

    if precision + recall > 0:
        return 2*precision*recall/(precision + recall)
    else:
        return 0.0


def f_measure_classic(labels_true, labels_pred):
    assert len(labels_pred) == len(labels_true)

    index = np.arange(len(labels_pred))

    index_product = np.dstack(np.meshgrid(index, index)).reshape(-1, 2)
    not_same = index_product[:, 0] != index_product[:, 1]

    labels_true_product = np.dstack(np.meshgrid(labels_true, labels_true)).reshape(-1, 2)[not_same]
    labels_pred_product = np.dstack(np.meshgrid(labels_pred, labels_pred)).reshape(-1, 2)[not_same]

    same_cluster_true = (labels_true_product[:, 0] == labels_true_product[:, 1])
    same_cluster_pred = (labels_pred_product[:, 0] == labels_pred_product[:, 1])

    tp = np.sum(same_cluster_true & same_cluster_pred)
    fp = np.sum(~same_cluster_true & same_cluster_pred)
    fn = np.sum(same_cluster_true & ~same_cluster_pred)

    precision = tp / (tp + fn)
    recall = tp / (tp + fp)

    if tp == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)


def jaccard_fast(labels_true, labels_pred):
    assert len(labels_pred) == len(labels_true)

    index = np.arange(len(labels_pred))

    index_product = np.dstack(np.meshgrid(index, index)).reshape(-1, 2)
    not_same = index_product[:, 0] != index_product[:, 1]

    labels_true_product = np.dstack(np.meshgrid(labels_true, labels_true)).reshape(-1, 2)[not_same]
    labels_pred_product = np.dstack(np.meshgrid(labels_pred, labels_pred)).reshape(-1, 2)[not_same]

    same_cluster_true = (labels_true_product[:, 0] == labels_true_product[:, 1])
    same_cluster_pred = (labels_pred_product[:, 0] == labels_pred_product[:, 1])

    tp = np.sum(same_cluster_true & same_cluster_pred)
    fp = np.sum(~same_cluster_true & same_cluster_pred)
    fn = np.sum(same_cluster_true & ~same_cluster_pred)

    return tp / (tp + fp + fn)


def jaccard_custom(labels_true, labels_pred):
    assert len(labels_pred) == len(labels_true)
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    index = np.arange(len(labels_pred))

    triplets = [(x, y, z) for x, y, z in zip(index, labels_true, labels_pred)]

    for triplet1, triplet2 in itertools.product(triplets, triplets):
        if triplet1[0] != triplet2[0]:
            if triplet1[1] == triplet2[1] and triplet1[2] == triplet2[2]:
                tp += 1
            elif triplet1[1] != triplet2[1] and triplet1[2] != triplet2[2]:
                tn += 1
            elif triplet1[1] != triplet2[1] and triplet1[2] == triplet2[2]:
                fp += 1
            else:
                fn += 1

    return tp / (tp + fp + fn)
