# -*- coding: utf-8 -*-
"""
Prevalence of the ranking of the methods over the draws of the Monte Carlo comparison.

Usage:  python ranking_montecarlo.py

In each draw the methods are ranked by their AMI with the true partition (1 = highest; ties get the average rank, and
a method tied for the highest AMI counts as first). Over the draws in which every method of the set is available, it
reports for each method: mean and median rank, share of draws in which it is first, share in which it is among the
first three, and share of draws in which it is above SDEC. Methods are ordered by median rank, which is less
sensitive than the mean to a few draws with a poor result; ties are broken by the share of draws in which the method
is first, and then by the mean rank. Two sets of methods:
  'una_por_metodo'  one output per method, as in the paper: SDEC, SH, SKATER, REDCAP, SCHC with Ward linkage,
                    PCA(6) + constrained Ward, the control smoothed twice, SDCN and the graph autoencoder with the
                    constrained step and with their own output (tuned configuration when it exists, reference
                    settings otherwise), Region2Vec, SBD, DTW and k-Shape, and Morelli et al. (Algorithm 1)
  'todas'           every output in the result files (except the oracle of Morelli et al., which is not a method)
Reads resultados/montecarlo_a<amplitude>.jsonl and, when they exist, montecarlo_profundos_*, montecarlo_series_*,
montecarlo_region2vec_* and montecarlo_ajustados_*. Writes resultados/ranking_<set>.csv,
resultados/ranking_victorias_<amplitude>.csv (share of draws in which the row method has a higher AMI than the column
method, set 'una_por_metodo') and figuras/ranking_montecarlo.png (distribution of ranks, set 'una_por_metodo').
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos

FUENTES = ('montecarlo_a[0-9]*', 'montecarlo_profundos_a*', 'montecarlo_series_a*', 'montecarlo_region2vec_a*',
           'montecarlo_ajustados_a*')
NOMBRES = {'sdec': 'SDEC', 'sh': 'SH', 'skater': 'SKATER', 'redcap': 'REDCAP', 'schc_ward': 'SCHC (Ward)',
           'pca6_ward': 'PCA(6) + c. Ward', 'suav2': 'Smoothed data + c. Ward', 'sdcn_ward': 'SDCN + c. Ward',
           'sdcn': 'SDCN', 'gae_ward': 'GAE + c. Ward', 'gae_kmeans': 'GAE + k-means', 'region2vec': 'Region2Vec',
           'sbd_ward': 'SBD + c. Ward', 'dtw_ward': 'DTW + c. Ward', 'kshape': 'k-Shape',
           'morelli': 'Morelli et al. (2025)'}
PRINCIPALES = list(NOMBRES)


def leer(patron):
    filas = []
    for ruta in sorted(glob.glob(os.path.join(constantes.DIR_RESULTADOS, patron + '.jsonl'))):
        filas += [json.loads(l) for l in open(ruta)]
    return filas


def tabla_ami(obj):
    largo = []
    for patron in FUENTES:
        for f in leer(patron):
            for m, g in f['grupos'].items():
                largo.append({'amplitud': f['amplitud'], 'replica': f['replica'], 'metodo': m, 'AMI': ami(obj, g)})
    largo = pd.DataFrame(largo).drop_duplicates(['amplitud', 'replica', 'metodo'], keep='last')
    return {a: d.pivot(index='replica', columns='metodo', values='AMI') for a, d in largo.groupby('amplitud')}


def una_por_metodo(t):
    """One output per method; the tuned configuration replaces the reference settings when it exists."""
    cols = {}
    for m in PRINCIPALES:
        if m + '_ajustado' in t.columns:
            cols[NOMBRES[m] + ' (tuned)'] = t[m + '_ajustado']
        elif m in t.columns:
            cols[NOMBRES[m]] = t[m]
    return pd.DataFrame(cols)


def ranking(t):
    t = t.dropna()
    R = np.apply_along_axis(lambda v: rankdata(-v, method='average'), 1, t.values)
    Rmin = np.apply_along_axis(lambda v: rankdata(-v, method='min'), 1, t.values)
    sdec = t['SDEC'].values if 'SDEC' in t.columns else t['sdec'].values
    res = pd.DataFrame({'rango_medio': R.mean(0), 'rango_mediano': np.median(R, 0),
                        'primero_pct': 100 * (Rmin == 1).mean(0), 'top3_pct': 100 * (Rmin <= 3).mean(0),
                        'sobre_sdec_pct': 100 * (t.values > sdec[:, None]).mean(0)}, index=t.columns)
    res = res.sort_values(['rango_mediano', 'primero_pct', 'rango_medio'], ascending=[True, False, True])
    return res, R, t


def figura(paneles, ruta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    azul = LinearSegmentedColormap.from_list('azul', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab',
                                                      '#0d366b'])
    fig, ejes = plt.subplots(1, len(paneles), figsize=(6.2 * len(paneles), 5.6), constrained_layout=True)
    ejes = np.atleast_1d(ejes)
    for ax, (titulo, res, R, t) in zip(ejes, paneles):
        orden = list(res.index)
        M = len(orden)
        frec = np.zeros((M, M))
        for i, m in enumerate(orden):
            j = list(t.columns).index(m)
            pos = np.clip(np.rint(R[:, j]).astype(int), 1, M)
            frec[i] = 100 * np.bincount(pos, minlength=M + 1)[1:] / len(R)
        im = ax.imshow(frec, cmap=azul, vmin=0, vmax=100, aspect='auto')
        for i in range(M):
            for j in range(M):
                if frec[i, j] >= 10:
                    ax.text(j, i, '%.0f' % frec[i, j], ha='center', va='center', fontsize=7,
                            color='white' if frec[i, j] >= 55 else '#1f1f1f')
        ax.set_xticks(range(M))
        ax.set_xticklabels([str(k) for k in range(1, M + 1)], fontsize=8)
        ax.set_yticks(range(M))
        ax.set_yticklabels(['%s  (%g; %.0f%%)' % (m, res.loc[m, 'rango_mediano'], res.loc[m, 'primero_pct'])
                            for m in orden], fontsize=8)
        ax.set_xlabel('Rank in the draw (1 = highest AMI)', fontsize=9)
        ax.set_title('%s, %d draws' % (titulo, len(R)), fontsize=10, loc='left')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    # the ordering of the rows and the meaning of the parentheses are explained in the caption of the figure in the paper
    cb = fig.colorbar(im, ax=ejes, shrink=0.6)
    cb.set_label('% of draws', fontsize=9)
    cb.outline.set_visible(False)
    fig.savefig(ruta, dpi=200, bbox_inches='tight')


def main():
    obj = datos.objetivo_simulacion(datos.departamentos())
    tablas = tabla_ami(obj)
    paneles = []
    for nombre, selector in (('una_por_metodo', una_por_metodo),
                             ('todas', lambda t: t.drop(columns=[c for c in t.columns if c == 'morelli_oraculo']))):
        salida = []
        for a, t in sorted(tablas.items()):
            res, R, tt = ranking(selector(t))
            res.insert(0, 'sorteos', len(tt))
            res.insert(0, 'amplitud', a)
            salida.append(res.reset_index().rename(columns={'index': 'metodo'}))
            print('\n%s, a = %g (%d draws with every method)' % (nombre, a, len(tt)))
            print(res.drop(columns=['amplitud', 'sorteos']).round(1).to_string())
            if nombre == 'una_por_metodo':
                paneles.append(('a = %g' % a, res, R, tt))
                v = pd.DataFrame({m: [100 * (tt[f] > tt[m]).mean() for f in tt.columns] for m in tt.columns},
                                 index=tt.columns).loc[res.index, res.index]
                v.round(1).to_csv(os.path.join(constantes.DIR_RESULTADOS, 'ranking_victorias_a%g.csv' % a))
        pd.concat(salida).to_csv(os.path.join(constantes.DIR_RESULTADOS, 'ranking_%s.csv' % nombre), index=False)
    os.makedirs(constantes.DIR_FIGURAS, exist_ok=True)
    figura(paneles, os.path.join(constantes.DIR_FIGURAS, 'ranking_montecarlo.png'))


if __name__ == '__main__':
    main()
