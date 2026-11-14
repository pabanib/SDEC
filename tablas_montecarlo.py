# -*- coding: utf-8 -*-
"""
Tables of the Monte Carlo comparison (resultados/montecarlo_a*.jsonl, written by correr_montecarlo.py, and, if they
exist, resultados/montecarlo_profundos_a*.jsonl, written by correr_montecarlo_profundos.py on the same draws).

Usage:  python tablas_montecarlo.py

For each amplitude of the noise it computes, against the target partition and over the replications:
  * mean, standard deviation, minimum and maximum of the adjusted mutual information (AMI) of every method, and the
    mean of the mutual information, the homogeneity and the Rand score;
  * the paired difference in AMI between SDEC and each method on the same draw of the data: mean, percentile bootstrap
    95% interval of the mean (9,999 resamples of the replications, seed 0) and number of replications in which SDEC
    is above the method;
  * the same paired comparison against 'mejor_otro', the highest AMI among all the other methods in each replication.
    This benchmark chooses the competitor after seeing the true partition, so it is favorable to the other methods.
    'mejor_otro' is taken among the methods of correr_montecarlo.py; when the deep graph-based methods are available,
    'mejor_otro_con_profundos' takes it among all the methods. The intervals of the deep methods use a separate
    random stream (seed 1), so the values of the other rows do not depend on whether those methods were run.
Writes resultados/montecarlo_metricas.csv and resultados/montecarlo_pareado.csv and prints both.
"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score as ami, mutual_info_score as mi, homogeneity_score as hom,
                             rand_score)
import constantes
from sdec import datos
from sdec.experimentos import leer_jsonl

METRICAS = [('AMI', ami), ('MI', mi), ('Homogeneity', hom), ('Rand', rand_score)]
B = 9999


def intervalo(d, rng):
    """Percentile bootstrap 95% interval of the mean of d."""
    medias = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(medias, [2.5, 97.5])


def main():
    obj = datos.objetivo_simulacion(datos.departamentos())
    filas = []
    for ruta in sorted(glob.glob(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_a[0-9]*.jsonl'))):
        filas += leer_jsonl(ruta)
    if not filas:
        raise SystemExit('No results found: run correr_montecarlo.py first.')
    base = {m for f in filas for m in f['grupos']}
    extra = {}
    for ruta in sorted(glob.glob(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_profundos_a*.jsonl'))):
        extra.update({(f['amplitud'], f['replica']): f['grupos'] for f in leer_jsonl(ruta)})
    for f in filas:
        f['grupos'].update(extra.get((f['amplitud'], f['replica']), {}))
    largo = []
    for f in filas:
        for m, g in f['grupos'].items():
            fila = {'amplitud': f['amplitud'], 'replica': f['replica'], 'metodo': m}
            fila.update({nm: fn(obj, g) for nm, fn in METRICAS})
            largo.append(fila)
    largo = pd.DataFrame(largo)

    res = largo.groupby(['amplitud', 'metodo']).agg(
        replicas=('AMI', 'size'), AMI_media=('AMI', 'mean'), AMI_sd=('AMI', 'std'), AMI_min=('AMI', 'min'),
        AMI_max=('AMI', 'max'), MI_media=('MI', 'mean'), Homog_media=('Homogeneity', 'mean'),
        Rand_media=('Rand', 'mean')).reset_index()
    res = res.sort_values(['amplitud', 'AMI_media'], ascending=[True, False])
    res.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_metricas.csv'), index=False)
    print(res.round(3).to_string(index=False))

    rng = np.random.default_rng(0)
    par = []
    for a, d in largo.groupby('amplitud'):
        t = d.pivot(index='replica', columns='metodo', values='AMI')
        otros = [c for c in t.columns if c != 'sdec' and c in base]
        t['mejor_otro'] = t[otros].max(axis=1)
        for m in otros + ['mejor_otro']:
            dif = (t['sdec'] - t[m]).dropna().values
            lo, hi = intervalo(dif, rng)
            par.append({'amplitud': a, 'metodo': m, 'replicas': len(dif), 'dif_media': dif.mean(), 'ic95_inf': lo,
                        'ic95_sup': hi, 'dif_min': dif.min(), 'sdec_arriba': int((dif > 0).sum())})
    rng = np.random.default_rng(1)
    for a, d in largo.groupby('amplitud'):
        t = d.pivot(index='replica', columns='metodo', values='AMI')
        profundos = [c for c in t.columns if c not in base]
        if not profundos:
            continue
        t = t.dropna(subset=profundos)                      # replications in which the deep methods were run
        t['mejor_otro_con_profundos'] = t.drop(columns='sdec').max(axis=1)
        for m in profundos + ['mejor_otro_con_profundos']:
            dif = (t['sdec'] - t[m]).values
            lo, hi = intervalo(dif, rng)
            par.append({'amplitud': a, 'metodo': m, 'replicas': len(dif), 'dif_media': dif.mean(), 'ic95_inf': lo,
                        'ic95_sup': hi, 'dif_min': dif.min(), 'sdec_arriba': int((dif > 0).sum())})
    par = pd.DataFrame(par).sort_values(['amplitud', 'dif_media'])
    par.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_pareado.csv'), index=False)
    print('\nPaired difference in AMI, SDEC minus each method (same draw of the data):')
    print(par.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
