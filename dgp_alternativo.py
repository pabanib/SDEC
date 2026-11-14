# -*- coding: utf-8 -*-
"""
Alternative data generating process, not used to develop or configure SDEC (out-of-design evaluation).

Same 525 departments, target partition and curves as dgp_simulacion.py, but the noise has a unit-level shift and a
period-by-period component with serial dependence and spatial autocorrelation:
    y_it^v = max{0, f_r(i)^v(t) + u_i^v + v_it^v},
    u_i^v ~ U(-b, b) with b chosen so that Var(u) = a^2 / 24,
    v_t^v = (I - lambda W_r)^{-1} e_t^v,  e_it^v = phi e_i,t-1^v + eta_it^v,  eta ~ N(0, 1), rescaled so that the
    variance of v over units and periods is a^2 / 24,
with lambda = 0.5 (W_r the row-standardized 6-nearest-neighbor matrix), phi = 0.7 and a = 0.5 or 1. The total noise
variance, a^2 / 12, equals that of the level shift of dgp_simulacion.py, so the two amplitudes are comparable with the
two scenarios of the paper. Each column of the n x 200 matrix is standardized across departments.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from dgp_simulacion import FUNC1, FUNC2

LAMBDA, PHI = 0.5, 0.7


def genera_datos_alt(obj, amplitud, semilla, W, periodos=100):
    rng = np.random.default_rng(semilla)
    n = len(obj)
    Wr = sp.csr_matrix(W).multiply(1.0 / sp.csr_matrix(W).sum(1)).tocsc()
    lu = splu((sp.eye(n, format='csc') - LAMBDA * Wr).tocsc())
    p = np.arange(periodos)
    b = np.sqrt(3.0 * amplitud ** 2 / 24.0)                  # Var(U(-b, b)) = b^2 / 3 = a^2 / 24
    var = []
    for FUNC in (FUNC1, FUNC2):
        u = rng.uniform(-b, b, n)
        e = np.zeros((n, periodos))
        e[:, 0] = rng.standard_normal(n) / np.sqrt(1 - PHI ** 2)
        for t in range(1, periodos):
            e[:, t] = PHI * e[:, t - 1] + rng.standard_normal(n)
        v = lu.solve(e)
        v *= np.sqrt(amplitud ** 2 / 24.0) / v.std()
        y = np.array([FUNC[r](p) for r in obj]) + u[:, None] + v
        var.append(y * (y > 0))
    m = np.c_[var[0], var[1]]
    return (m - m.mean(0)) / m.std(0)


def objetivo_aleatorio(W, K, semilla):
    """Random target partition into K regions that are connected in the neighbor graph, grown from K random seed
    departments with unequal growth rates (so the regions have unequal sizes and irregular shapes). Each region is then
    given one of the six curves of Table 1 at random (labels 1..6). With K = 6, a partition of the centroids by Ward's
    method, which ignores the data, recovers these targets with a mean AMI of about 0.50, against 0.667 for the
    macro-regions of the paper."""
    import scipy.sparse as sp
    rng = np.random.default_rng(semilla)
    A = (sp.csr_matrix(W) + sp.csr_matrix(W).T > 0).tocsr()
    n = A.shape[0]
    lab = -np.ones(n, int)
    semillas = rng.choice(n, K, replace=False)
    lab[semillas] = np.arange(K)
    tasa = rng.uniform(0.3, 3.0, K)
    frontera = [set(A[s].indices) - {s} for s in semillas]
    while (lab < 0).any():
        activos = np.array([any(lab[j] < 0 for j in frontera[k]) for k in range(K)])
        if not activos.any():
            break
        p = tasa * activos
        k = rng.choice(K, p=p / p.sum())
        cand = sorted(j for j in frontera[k] if lab[j] < 0)
        j = cand[rng.integers(len(cand))]
        lab[j] = k
        frontera[k] |= set(A[j].indices)
    curvas = rng.permutation(np.arange(1, 7))[:K]
    return curvas[lab]
