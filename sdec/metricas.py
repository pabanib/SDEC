# -*- coding: utf-8 -*-
"""Metrics used in the paper."""
import numpy as np
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score)


def rbtss(X, g):
    """RBTSS = 1 - WSS/TSS, with the group mean as center and all observations."""
    X = np.asarray(X, float)
    g = np.asarray(g)
    tss = ((X - X.mean(0)) ** 2).sum()
    wss = sum(((X[g == k] - X[g == k].mean(0)) ** 2).sum() for k in np.unique(g))
    return float(1.0 - wss / tss)


def piso_aleatorio(X, g, B=200, semilla=0):
    """Expected RBTSS of random partitions with the same group sizes."""
    rng = np.random.default_rng(semilla)
    return float(np.mean([rbtss(X, rng.permutation(g)) for _ in range(B)]))


def coseno_medio(A, B):
    """Mean cosine similarity between each row of A and the corresponding row of B."""
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return float(np.mean(num / den))


def intrinsecas(X, g):
    return {'silueta': float(silhouette_score(X, g)),
            'calinski_harabasz': float(calinski_harabasz_score(X, g)),
            'davies_bouldin': float(davies_bouldin_score(X, g))}


def resumen_corrida(sc, prop_st, T):
    """Summary of one run of SDEC on the standardized data prop_st (n x 2T)."""
    X1, X2 = prop_st[:, :T], prop_st[:, T:]
    g = np.asarray(sc.y_pred)
    emb = sc.encoder.predict([X1, X2], verbose=0)
    pred = sc.autoencoder.predict([X1, X2], verbose=0)
    return {
        'n_grupos': int(len(np.unique(g))),
        'tamanos': sorted(int(x) for x in np.bincount(g)),
        'ultima_actualizacion': sc.ultima_actualizacion,
        'coseno_medio': coseno_medio(pred, prop_st),
        'rbtss_original': rbtss(prop_st, g),
        'rbtss_encoder': rbtss(emb, g),
        'piso_aleatorio_original': piso_aleatorio(prop_st, g),
        'intrinsecas_encoder': intrinsecas(emb, g),
    }
