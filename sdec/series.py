# -*- coding: utf-8 -*-
"""
Comparison methods that treat each variable of an area as a time series, and the spatiotemporal hierarchical
clustering of Chavent et al. (2018) and Morelli et al. (2025).

dtw_matriz      DTW dissimilarity between the series of one variable, with local cost |x_s - y_r| and the symmetric
                step pattern (diagonal step weighted 2), the defaults of TSclust::diss.DTWARP / dtw::dtw used by
                Morelli et al. (2025); no window.
sbd_matriz      shape-based distance of k-Shape (Paparrizos and Gravano, 2015): 1 - max normalized cross-correlation
                of the z-normalized series.
ward_like       Ward-like agglomerative clustering on dissimilarities (Chavent et al., 2018, ClustGeo::hclustgeo:
                Lance-Williams update of ward.D applied to delta_ij = w_i w_j / (w_i + w_j) d_ij^2), optionally
                restricted to merges of groups that are adjacent in a graph (strict contiguity constraint).
morelli         ClustGeo with several dissimilarity matrices (one DTW matrix per variable and the geographic distance
                matrix), each divided by its maximum, combined with weights on the simplex; the weights are chosen
                with Algorithm 1 of Morelli et al. (2025): maximize the explained pseudo-inertia summed over the
                matrices, without using the true partition.
"""
import numpy as np
import numba
import scipy.sparse as sp


@numba.njit(cache=True)
def _dtw(x, y):
    n, m = x.shape[0], y.shape[0]
    g = np.full((n + 1, m + 1), np.inf)
    g[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(x[i - 1] - y[j - 1])
            a = g[i - 1, j - 1] + 2.0 * d
            b = g[i - 1, j] + d
            c = g[i, j - 1] + d
            g[i, j] = min(a, min(b, c))
    return g[n, m]


@numba.njit(parallel=True, cache=True)
def dtw_matriz(S):
    """S: (n, T) series of one variable. Returns the n x n DTW dissimilarity matrix."""
    n = S.shape[0]
    D = np.zeros((n, n))
    for i in numba.prange(n):
        for j in range(i + 1, n):
            D[i, j] = _dtw(S[i], S[j])
    return D + D.T


def sbd_matriz(S):
    """S: (n, T). Shape-based distance between z-normalized series, via FFT cross-correlation."""
    Z = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    n, T = Z.shape
    L = 1 << int(np.ceil(np.log2(2 * T - 1)))
    F = np.fft.rfft(Z, L, axis=1)
    nrm = np.linalg.norm(Z, axis=1)
    D = np.zeros((n, n))
    for i in range(n):
        cc = np.fft.irfft(F[i][None, :] * np.conj(F), L, axis=1)
        D[i] = 1.0 - cc.max(1) / (nrm[i] * nrm)
    D = np.maximum((D + D.T) / 2, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def wardinit(D, w=None):
    """delta_ij = w_i w_j / (w_i + w_j) d_ij^2 (ClustGeo::wardinit), uniform weights 1/n by default."""
    n = D.shape[0]
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float)
    return (np.outer(w, w) / (w[:, None] + w[None, :])) * D ** 2


def ward_like(delta, K, w=None, conectividad=None):
    """Agglomerative Ward-like clustering (hclust 'ward.D' on delta, members = w) cut at K groups.
    conectividad: sparse symmetric adjacency; if given, only adjacent groups can merge (the group graph is the union
    of the adjacencies of its members). Returns labels 0..K-1."""
    n = delta.shape[0]
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float).copy()
    d = delta.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    activo = np.ones(n, bool)
    etiqueta = np.arange(n)
    if conectividad is not None:
        A = (sp.csr_matrix(conectividad) + sp.csr_matrix(conectividad).T).toarray() > 0
        np.fill_diagonal(A, False)
    else:
        A = None
    for _ in range(n - K):
        dd = d if A is None else np.where(A, d, np.inf)
        dd = np.where(activo[:, None] & activo[None, :], dd, np.inf)
        k = int(np.argmin(dd))
        i, j = divmod(k, n)
        if not np.isfinite(dd[i, j]):
            break                                  # the graph has more than K components
        wi, wj, wk = w[i], w[j], w
        nuevo = ((wi + wk) * d[i] + (wj + wk) * d[j] - wk * d[i, j]) / (wi + wj + wk)
        d[i, :] = nuevo
        d[:, i] = nuevo
        d[i, i] = np.inf
        d[j, :] = np.inf
        d[:, j] = np.inf
        w[i] = wi + wj
        activo[j] = False
        etiqueta[etiqueta == j] = i
        if A is not None:
            A[i] = A[i] | A[j]
            A[:, i] = A[i]
            A[i, i] = False
            A[j, :] = False
            A[:, j] = False
    return np.unique(etiqueta, return_inverse=True)[1]


def pseudo_inercia_intra(D, g, w=None):
    """Within-group pseudo-inertia (ClustGeo::withindiss): sum over groups of sum_{i<j} w_i w_j / (2 mu_k) d_ij^2."""
    n = D.shape[0]
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float)
    tot = 0.0
    for k in np.unique(g):
        m = np.where(g == k)[0]
        mu = w[m].sum()
        tot += (np.outer(w[m], w[m]) * D[np.ix_(m, m)] ** 2).sum() / (2.0 * 2.0 * mu)
    return tot


def simplex(p, paso=0.1):
    """Weights on the simplex of p components with the given step."""
    m = int(round(1 / paso))
    out = []

    def rec(pref, resto, q):
        if q == 1:
            out.append(pref + [resto])
            return
        for a in range(resto + 1):
            rec(pref + [a], resto - a, q - 1)
    rec([], m, p)
    return np.array(out, float) / m


def morelli(Ds, K, paso=0.1, conectividad=None):
    """Ds: list of dissimilarity matrices (the last one geographic). Each is divided by its maximum, the Ward-like
    aggregation measures are combined with weights alpha on the simplex, and the partition into K groups is computed
    for each alpha. Algorithm 1 of Morelli et al. (2025): alpha* maximizes
        Qbar = 1 - sum_p W_p(P_K^alpha) / sum_p W_p(P_1).
    Returns (labels at alpha*, alpha*, list of (alpha, labels, Qbar))."""
    Dn = [D / D.max() for D in Ds]
    deltas = [wardinit(D) for D in Dn]
    T = [pseudo_inercia_intra(D, np.zeros(D.shape[0], int)) for D in Dn]
    rejilla = []
    for alfa in simplex(len(Ds), paso):
        delta = sum(a * dl for a, dl in zip(alfa, deltas))
        g = ward_like(delta, K, conectividad=conectividad)
        q = 1.0 - sum(pseudo_inercia_intra(D, g) for D in Dn) / sum(T)
        rejilla.append((alfa, g, q))
    mejor = max(rejilla, key=lambda r: r[2])
    return mejor[1], mejor[0], rejilla
