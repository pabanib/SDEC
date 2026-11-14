# -*- coding: utf-8 -*-
"""
Region2Vec (sdec/region2vec.py), with its reference settings, on the draws of the Monte Carlo comparison.

Usage:  python correr_montecarlo_region2vec.py <amplitude> [replications] [first replication]

Replication r uses the data of correr_montecarlo.py (seed SEMILLA_BASE + r), K = 6, the same 6-nearest-neighbor matrix
and the seed r for the network. One row per replication in resultados/montecarlo_region2vec_a<amplitude>.jsonl.
"""
import sys
import os
import json
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos
from sdec.region2vec import region2vec
from dgp_simulacion import genera_datos

SEMILLA_BASE = 100000      # same as correr_montecarlo.py


def leer_jsonl(ruta):
    return [json.loads(l) for l in open(ruta)] if os.path.exists(ruta) else []


def agregar_jsonl(ruta, fila):
    with open(ruta, 'a') as f:
        f.write(json.dumps(fila) + '\n')


def main():
    a = float(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    primera = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_region2vec_a%g.jsonl' % a)
    hechas = {f['replica'] for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if r in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        out = region2vec(X, W, K, r)
        agregar_jsonl(archivo, {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r,
                                'epocas': int(out['region2vec_epocas']),
                                'grupos': {'region2vec': [int(v) for v in out['region2vec']]}})
        print('a', a, 'replication', r, 'epochs', out['region2vec_epocas'], 'AMI',
              round(float(ami(obj, out['region2vec'])), 3), flush=True)


if __name__ == '__main__':
    main()
