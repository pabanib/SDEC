# -*- coding: utf-8 -*-
"""
Paired comparison of SDEC with the methods added in the second round, on the draws of the Monte Carlo comparison.

Usage:  python tablas_montecarlo_nuevos.py

Reads the labels of SDEC from resultados/montecarlo_a*.jsonl and, when they exist, those of
  montecarlo_series_a*.jsonl       DTW + constrained Ward, SBD + constrained Ward, k-Shape, Morelli et al. (2025)
  montecarlo_region2vec_a*.jsonl   Region2Vec
  montecarlo_ajustados_a*.jsonl    SDCN, graph autoencoder and Region2Vec with the tuned configurations
  montecarlo_profundos_a*.jsonl    SDCN and graph autoencoder with the reference settings (for the best-of-all row)
The paired comparison also includes two benchmarks that are not methods: the highest AMI among all the other
methods in each draw ('mejor_de_todos') and the highest among those that are not deep graph-based methods
('mejor_sin_profundos'); both are chosen after observing the true partition. For Morelli et al. it adds 'morelli_oraculo', the best AMI over the grid of weights in each draw (chosen with the true
partition, favorable to the method), the share of its regions that are connected in the neighbor graph, and the
weights selected by its Algorithm 1. Same statistics as tablas_montecarlo.py (paired difference, percentile bootstrap
95% interval with 9,999 resamples, seed 2, number of draws with SDEC above). Writes
resultados/montecarlo_nuevos_metricas.csv and resultados/montecarlo_nuevos_pareado.csv.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos

B = 9999
FUENTES = ('montecarlo_series', 'montecarlo_region2vec', 'montecarlo_ajustados', 'montecarlo_profundos')


def leer(patron):
    filas = []
    for ruta in sorted(glob.glob(os.path.join(constantes.DIR_RESULTADOS, patron))):
        filas += [json.loads(l) for l in open(ruta)]
    return filas


def intervalo(d, rng):
    medias = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(medias, [2.5, 97.5])


def conectadas(A, g):
    g = np.asarray(g)
    tot = 0
    for k in np.unique(g):
        m = np.where(g == k)[0]
        tot += connected_components(A[m][:, m])[0] == 1
    return tot / len(np.unique(g))


def main():
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    A = ((W + W.T) > 0).astype(int).tocsr()
    base = {(f['amplitud'], f['replica']): f for f in leer('montecarlo_a[0-9]*.jsonl')}
    largo, extra = [], []
    nuevos = set()
    for fuente in FUENTES:
        for f in leer(fuente + '_a*.jsonl'):
            clave = (f['amplitud'], f['replica'])
            if clave not in base:
                continue
            for m, g in f['grupos'].items():
                largo.append({'amplitud': f['amplitud'], 'replica': f['replica'], 'metodo': m,
                              'AMI': ami(obj, g), 'fuente': fuente})
                if fuente != 'montecarlo_profundos':
                    nuevos.add(m)
            if fuente == 'montecarlo_series':
                mejor = max(r[-1] for r in f['morelli_rejilla_ami'])
                largo.append({'amplitud': f['amplitud'], 'replica': f['replica'], 'metodo': 'morelli_oraculo',
                              'AMI': mejor, 'fuente': fuente})
                nuevos.add('morelli_oraculo')
                extra.append({'amplitud': f['amplitud'], 'replica': f['replica'],
                              'alfa_var1': f['morelli_alfa'][0], 'alfa_var2': f['morelli_alfa'][1],
                              'alfa_geo': f['morelli_alfa'][2],
                              'morelli_regiones_conectadas': conectadas(A, f['grupos']['morelli'])})
    for (a, r), f in base.items():
        largo.append({'amplitud': a, 'replica': r, 'metodo': 'sdec', 'AMI': ami(obj, f['grupos']['sdec']),
                      'fuente': 'montecarlo'})
        for m, g in f['grupos'].items():
            if m != 'sdec':
                largo.append({'amplitud': a, 'replica': r, 'metodo': m, 'AMI': ami(obj, g), 'fuente': 'montecarlo'})
    largo = pd.DataFrame(largo)
    res = largo[largo.metodo.isin(nuevos | {'sdec'})].groupby(['amplitud', 'metodo']).AMI.agg(
        ['size', 'mean', 'std', 'min', 'median', 'max']).reset_index().sort_values(['amplitud', 'mean'],
                                                                                    ascending=[True, False])
    res.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_nuevos_metricas.csv'), index=False)
    print(res.round(3).to_string(index=False))
    if extra:
        ex = pd.DataFrame(extra)
        print('\nMorelli et al.: weights selected by Algorithm 1 and share of regions connected in the neighbor graph')
        print(ex.drop(columns='replica').groupby('amplitud').agg(['mean', 'min', 'max']).round(3).T.to_string())
    rng = np.random.default_rng(2)
    rng_sin = np.random.default_rng(3)
    par = []
    for a, d in largo.groupby('amplitud'):
        t = d.drop_duplicates(['replica', 'metodo']).pivot(index='replica', columns='metodo', values='AMI')
        presentes = [m for m in sorted(nuevos) if m in t.columns]
        t = t.dropna(subset=presentes)
        reales = [c for c in t.columns if c not in ('sdec', 'morelli_oraculo')]
        t['mejor_de_todos'] = t[reales].max(axis=1)
        profundos = [c for c in reales if c.startswith(('gae', 'sdcn', 'region2vec'))]
        t['mejor_sin_profundos'] = t[[c for c in reales if c not in profundos]].max(axis=1)
        for m in presentes + ['mejor_de_todos', 'mejor_sin_profundos']:
            dif = (t['sdec'] - t[m]).values
            # separate generator for the benchmark added last, so that the intervals of the other rows do not change
            lo, hi = intervalo(dif, rng_sin if m == 'mejor_sin_profundos' else rng)
            par.append({'amplitud': a, 'metodo': m, 'replicas': len(dif), 'media': t[m].mean(), 'de': t[m].std(),
                        'dif_media': dif.mean(), 'ic95_inf': lo,
                        'ic95_sup': hi, 'sdec_arriba': int((dif > 0).sum())})
    par = pd.DataFrame(par).sort_values(['amplitud', 'dif_media'])
    par.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_nuevos_pareado.csv'), index=False)
    print('\nPaired difference in AMI, SDEC minus each method (same draw):')
    print(par.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
