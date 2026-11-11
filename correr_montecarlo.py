# -*- coding: utf-8 -*-
"""
Monte Carlo comparison over draws of the simulated data (data generating process of Section 4).

Usage:  python correr_montecarlo.py <amplitude> [replications] [first replication]
        python correr_montecarlo.py 0.5 100          scenario 1 (a = 0.5), replications 1 to 100
        python correr_montecarlo.py 1.0 100          scenario 2 (a = 1)
        python correr_montecarlo.py 1.0 50 101       replications 101 to 150

In replication r the data are drawn again with seed SEMILLA_BASE + r (dgp_simulacion.genera_datos; target partition,
curves and standardization as in the paper) and every method is applied to that same matrix, with K = 6 regions and
the same 6-nearest-neighbor matrix:
  sdec        : SDEC, one initialization (seed r), exactly as in correr_simulaciones.py
  sh          : Ward constrained by the neighbor graph, on the data
  pca6_ward   : constrained Ward on the first six principal components
  suav1..3    : constrained Ward on the data smoothed 1, 2 or 3 times with the row-standardized neighbor matrix
                (control without neural network of control_suavizado.py)
  skater, redcap, schc_<linkage> : pygeoda (same library as the R package rgeoda), if it is installed
One row per replication, with the labels of every method, is appended to resultados/montecarlo_a<amplitude>.jsonl (one
file per amplitude, so the two scenarios can run at the same time in two terminals). Replications already in the file
are skipped, so an interrupted run resumes where it stopped. The metrics and the paired comparison are computed by
tablas_montecarlo.py. Each replication takes about as long as one SDEC run.
"""
import sys
import os
import numpy as np
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos
from sdec.experimentos import entrenar, leer_jsonl, agregar_jsonl
from dgp_simulacion import genera_datos

try:
    import pygeoda
except ImportError:
    pygeoda = None

SEMILLA_BASE = 100000                    # data seed of replication r: SEMILLA_BASE + r
ENLACES_SCHC = ('single', 'complete', 'average', 'ward')


def deterministas(X, W, Wr, K):
    """Methods without random initialization that only need the data and the neighbor matrix."""
    ward = lambda Z: AgglomerativeClustering(K, connectivity=W).fit(Z).labels_
    lab = {'sh': ward(X), 'pca6_ward': ward(PCA(6).fit_transform(X))}
    Z = X.copy()
    for m in (1, 2, 3):
        Z = Wr @ Z
        lab['suav%d' % m] = ward(Z)
    return lab


def de_geoda(X, dep, K, k):
    """SKATER, REDCAP and SCHC (pygeoda) on the data X, with a k-nearest-neighbor matrix."""
    cols = [str(i) for i in range(X.shape[1])]
    g = gpd.GeoDataFrame(X, columns=cols, geometry=dep.geometry.values, crs=dep.crs)
    w = pygeoda.knn_weights(pygeoda.open(g), k)
    d = g[cols]
    lab = {'skater': pygeoda.skater(K, w, d)['Clusters'],
           'redcap': pygeoda.redcap(K, w, d, 'fullorder-averagelinkage')['Clusters']}
    for e in ENLACES_SCHC:
        lab['schc_' + e] = pygeoda.schc(K, w, d, e)['Clusters']
    return lab


def main():
    a = float(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    primera = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    Wr = W.multiply(1 / W.sum(1)).tocsr()
    if pygeoda is None:
        print('pygeoda not found: SKATER, REDCAP and SCHC are skipped (pip install pygeoda).', flush=True)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_a%g.jsonl' % a)
    hechas = {(f['amplitud'], f['replica']) for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if (a, r) in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        T = X.shape[1] // 2
        lab = deterministas(X, W, Wr, K)
        if pygeoda is not None:
            lab.update(de_geoda(X, dep, K, k))
        sc, w0 = entrenar(X, K, T, r)
        sc.ajustar_modelo([X[:, :T], X[:, T:]], W, semilla=r)
        lab['sdec'] = sc.y_pred
        fila = {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r, 'semilla_sdec': r,
                'iteraciones': int(sc.n_iter),
                'grupos': {m: [int(v) for v in np.asarray(g)] for m, g in lab.items()}}
        agregar_jsonl(archivo, fila)
        print('a', a, 'replication', r, 'AMI', {m: round(float(ami(obj, np.asarray(g))), 3) for m, g in lab.items()},
              flush=True)


if __name__ == '__main__':
    main()
