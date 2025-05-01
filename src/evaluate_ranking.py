import numpy as np
import pandas as pd

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores)[:k]
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2))) if relevance_scores.size else 0.0

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    ideal = sorted(y_true, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg_at_k(y_true_sorted, k) / ideal_dcg if ideal_dcg else 0.0

def precision_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    return np.mean(y_true_sorted[:k])

def evaluate_grouped_metrics(results_df, k=5):
    grouped = results_df.groupby('query_id')
    ndcg_scores, precision_scores = [], []

    for _, group in grouped:
        if group['true_relevance'].sum() == 0:
            continue
        ndcg = ndcg_at_k(group['true_relevance'].values, group['pred_proba'].values, k)
        precision = precision_at_k(group['true_relevance'].values, group['pred_proba'].values, k)
        ndcg_scores.append(ndcg)
        precision_scores.append(precision)

    return np.mean(ndcg_scores), np.mean(precision_scores)