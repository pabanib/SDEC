# -*- coding: utf-8 -*-
"""
Deep graph-based comparison methods on the draws of the Monte Carlo comparison (correr_montecarlo.py).

Usage:  python correr_montecarlo_profundos.py <amplitude> [replications] [first replication]
        python correr_montecarlo_profundos.py 0.5 100
        python correr_montecarlo_profundos.py 1.0 100

Replication r uses the same data as correr_montecarlo.py (seed SEMILLA_BASE + r), the same K = 6 and the same
6-nearest-neighbor matrix, and the seed r for the networks. Methods (sdec/profundos.py):
  sdcn, sdcn_q, sdcn_ward : Structural Deep Clustering Network (Bo et al., 2020), with the series as node features
                            and the neighbor matrix as graph: output of the GCN branch, output of the autoencoder
                            branch, and Ward's method constrained by the neighbor graph applied to the GCN output
  gae_kmeans, gae_ward    : graph autoencoder (Kipf and Welling, 2016) followed by k-means or by constrained Ward
One row per replication is appended to resultados/montecarlo_profundos_a<amplitude>.jsonl; replications already in
the file are skipped. tablas_montecarlo.py adds these methods to the tables of the Monte Carlo comparison.
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos, profundos
from sdec.experimentos import leer_jsonl, agregar_jsonl
from dgp_simulacion import genera_datos
from correr_montecarlo import SEMILLA_BASE


def main():
    a = float(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    primera = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_profundos_a%g.jsonl' % a)
    hechas = {(f['amplitud'], f['replica']) for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if (a, r) in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        lab = profundos.sdcn(X, W, K, r)
        lab.update(profundos.gae(X, W, K, r))
        fila = {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r, 'semilla_redes': r,
                'grupos': {m: [int(v) for v in np.asarray(g)] for m, g in lab.items()}}
        agregar_jsonl(archivo, fila)
        print('a', a, 'replication', r, 'AMI', {m: round(float(ami(obj, np.asarray(g))), 3) for m, g in lab.items()},
              flush=True)


if __name__ == '__main__':
    main()
