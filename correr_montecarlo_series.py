# -*- coding: utf-8 -*-
"""
Methods that treat the variables as time series, and the spatiotemporal hierarchical clustering of Chavent et al.
(2018) / Morelli et al. (2025), on the draws of the Monte Carlo comparison (correr_montecarlo.py).

Usage:  python correr_montecarlo_series.py <amplitude> [replications] [first replication]

Replication r uses the same data as correr_montecarlo.py (seed SEMILLA_BASE + r), K = 6 and the same 6-nearest-neighbor
matrix W; the two variables are the first and last 100 columns of the standardized matrix. Methods (sdec/series.py):
  dtw_ward     DTW matrix of each variable (divided by its maximum), equal weights, Ward-like clustering restricted to
               merges of groups adjacent in W (same strict constraint as the final step of SDEC)
  sbd_ward     the same with the shape-based distance of k-Shape
  kshape       k-Shape (tslearn, 5 initializations, seed r) on the z-normalized bivariate series, without spatial
               constraint
  morelli      ClustGeo with the two DTW matrices and the matrix of distances between the centroids of the departments,
               weights chosen by Algorithm 1 of Morelli et al. (2025) on a grid of step 0.1 on the simplex, without the
               true partition (soft spatial constraint, as in the original method)
  morelli_ward the weights chosen for morelli, with the strict constraint of W
For morelli the AMI with the true partition is also stored for every weight of the grid (evaluation only), so that
the best weight chosen with knowledge of the true partition can be reported as a benchmark favorable to the method.
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import constantes
from sdec import datos, series
from dgp_simulacion import genera_datos
import json

SEMILLA_BASE = 100000      # same as correr_montecarlo.py (not imported, to avoid loading TensorFlow)


def leer_jsonl(ruta):
    return [json.loads(l) for l in open(ruta)] if os.path.exists(ruta) else []


def agregar_jsonl(ruta, fila):
    with open(ruta, 'a') as f:
        f.write(json.dumps(fila) + '\n')


def metodos(X, W, Dgeo, K, semilla, obj=None):
    T = X.shape[1] // 2
    V = [np.ascontiguousarray(X[:, :T]), np.ascontiguousarray(X[:, T:])]
    lab, extra = {}, {}
    Ddtw = [series.dtw_matriz(v) for v in V]
    Dsbd = [series.sbd_matriz(v) for v in V]
    for nombre, Ds in (('dtw_ward', Ddtw), ('sbd_ward', Dsbd)):
        delta = sum(0.5 * series.wardinit(D / D.max()) for D in Ds)
        lab[nombre] = series.ward_like(delta, K, conectividad=W)
    Z = TimeSeriesScalerMeanVariance().fit_transform(np.stack(V, axis=2))
    lab['kshape'] = KShape(K, n_init=5, random_state=semilla).fit_predict(Z)
    g, alfa, rejilla = series.morelli(Ddtw + [Dgeo], K)
    lab['morelli'] = g
    Dn = [D / D.max() for D in Ddtw + [Dgeo]]
    lab['morelli_ward'] = series.ward_like(sum(a * series.wardinit(D) for a, D in zip(alfa, Dn)), K, conectividad=W)
    extra['morelli_alfa'] = [float(a) for a in alfa]
    if obj is not None:
        extra['morelli_rejilla_ami'] = [[float(a) for a in r[0]] + [float(ami(obj, r[1]))] for r in rejilla]
    return lab, extra


def main():
    a = float(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    primera = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    c = np.c_[dep.geometry.centroid.x, dep.geometry.centroid.y]
    Dgeo = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(2))
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_series_a%g.jsonl' % a)
    hechas = {(f['amplitud'], f['replica']) for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if (a, r) in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        lab, extra = metodos(X, W, Dgeo, K, r, obj)
        fila = {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r,
                'grupos': {m: [int(v) for v in np.asarray(g)] for m, g in lab.items()}}
        fila.update(extra)
        agregar_jsonl(archivo, fila)
        print('a', a, 'replication', r, 'alpha', extra['morelli_alfa'], 'AMI',
              {m: round(float(ami(obj, np.asarray(g))), 3) for m, g in lab.items()}, flush=True)


if __name__ == '__main__':
    main()
