# -*- coding: utf-8 -*-
"""
Monte Carlo comparison on the alternative data generating process (dgp_alternativo.py), which was not used to develop
or configure SDEC: noise with a unit-level shift plus a period-by-period component with serial dependence (AR(1),
phi = 0.7) and spatial autocorrelation (SAR, lambda = 0.5).

Usage:  python correr_montecarlo_alternativo.py <amplitude> <target> [replications] [first replication]
        python correr_montecarlo_alternativo.py 0.5 aleatorio 100
        python correr_montecarlo_alternativo.py 1.0 aleatorio 100
<target>: 'aleatorio' draws in each replication a new random partition into six regions connected in the neighbor
graph (dgp_alternativo.objetivo_aleatorio), so the target is not the one used to develop SDEC; 'macro' keeps the
macro-regions of the paper and changes only the noise.

Every method is applied to the same data in each draw (seed 200000 + r), with K = 6 and the same 6-nearest-neighbor
matrix, and with the settings used in Table 2 (no configuration is changed for this process): SDEC (one
initialization, seed r), SH, PCA(6) + constrained Ward, the smoothing controls, SKATER, REDCAP and SCHC (pygeoda),
SDCN and the graph autoencoder (reference settings), Region2Vec, and the time-series methods and Morelli et al.
(2025) of correr_montecarlo_series.py, and 'geografia', Ward's method on the coordinates of the centroids, which does
not use the data (the AMI that geography alone attains). One row per replication, with the target partition, in
resultados/montecarlo_alternativo_<target>_a<amplitude>.jsonl.
Each replication takes about as long as one SDEC run plus one SDCN run (5 to 8 minutes on a laptop CPU).
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos, profundos
from sdec.region2vec import region2vec
from sdec.experimentos import entrenar, leer_jsonl, agregar_jsonl
from sklearn.cluster import AgglomerativeClustering
from dgp_alternativo import genera_datos_alt, objetivo_aleatorio
from correr_montecarlo import deterministas, de_geoda, pygeoda
from correr_montecarlo_series import metodos as metodos_series

SEMILLA_BASE_ALT = 200000


def main():
    a = float(sys.argv[1])
    tipo = sys.argv[2]
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    primera = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    macro = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    Wr = W.multiply(1 / W.sum(1)).tocsr()
    c = np.c_[dep.geometry.centroid.x, dep.geometry.centroid.y]
    Dgeo = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(2))
    geografia = AgglomerativeClustering(K, connectivity=W, linkage='ward').fit_predict(c)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_alternativo_%s_a%g.jsonl' % (tipo, a))
    hechas = {f['replica'] for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if r in hechas:
            continue
        obj = objetivo_aleatorio(W, K, SEMILLA_BASE_ALT + r) if tipo == 'aleatorio' else macro
        X = genera_datos_alt(obj, a, SEMILLA_BASE_ALT + r, W)
        T = X.shape[1] // 2
        lab = deterministas(X, W, Wr, K)
        lab['geografia'] = geografia
        if pygeoda is not None:
            lab.update(de_geoda(X, dep, K, k))
        lab.update(profundos.sdcn(X, W, K, r))
        lab.update(profundos.gae(X, W, K, r))
        lab['region2vec'] = region2vec(X, W, K, r)['region2vec']
        ls, extra = metodos_series(X, W, Dgeo, K, r, obj)
        lab.update(ls)
        sc, w0 = entrenar(X, K, T, r)
        sc.ajustar_modelo([X[:, :T], X[:, T:]], W, semilla=r)
        lab['sdec'] = sc.y_pred
        fila = {'amplitud': a, 'objetivo_tipo': tipo, 'objetivo': [int(v) for v in obj], 'replica': r, 'semilla_datos': SEMILLA_BASE_ALT + r, 'semilla_redes': r,
                'iteraciones_sdec': int(sc.n_iter), 'morelli_alfa': extra['morelli_alfa'],
                'grupos': {m: [int(v) for v in np.asarray(g)] for m, g in lab.items()}}
        agregar_jsonl(archivo, fila)
        print('a', a, 'replication', r, 'AMI', {m: round(float(ami(obj, np.asarray(g))), 3) for m, g in lab.items()},
              flush=True)


if __name__ == '__main__':
    main()
