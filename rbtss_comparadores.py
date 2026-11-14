# -*- coding: utf-8 -*-
"""
RBTSS on the original variables of the partitions of the comparison methods in the two applications, with the same
data, the same number of regions K and the same k-nearest-neighbor matrix as SDEC (reference values for the RBTSS
of SDEC reported in Section 5).

Usage:  python rbtss_comparadores.py            writes resultados/rbtss_comparadores.csv

Methods (all deterministic given the data):
  sh          Ward constrained by the neighbor graph on the standardized variables
  pca6_ward   constrained Ward on the first six principal components (the dimension of the latent space of SDEC)
  skater, redcap, schc_ward : pygeoda with the same k-nearest-neighbor matrix (skipped if pygeoda is missing)
  kmeans      k-means without spatial constraint (20 initializations), an upper reference: it is the unconstrained
              optimum of the RBTSS up to the local search of k-means
  aleatorio   mean RBTSS of random partitions with the group sizes of SH (lower reference)
The RBTSS is computed on the standardized matrix that enters every method (prop_st), with metricas.rbtss.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import constantes
from sdec import datos, metricas

try:
    import pygeoda
except ImportError:
    pygeoda = None


def particiones(d, K, k):
    X, W = d['prop_st'], d['W']
    ward = lambda Z: AgglomerativeClustering(K, connectivity=W).fit(Z).labels_
    lab = {'sh': ward(X), 'pca6_ward': ward(PCA(6).fit_transform(X)),
           'kmeans': KMeans(K, n_init=20, random_state=0).fit_predict(X)}
    if pygeoda is not None:
        cols = [str(i) for i in range(X.shape[1])]
        g = gpd.GeoDataFrame(X, columns=cols, geometry=d['geo'].values, crs=d['geo'].crs)
        w = pygeoda.knn_weights(pygeoda.open(g), k)
        df = g[cols]
        lab['skater'] = np.asarray(pygeoda.skater(K, w, df)['Clusters'])
        lab['redcap'] = np.asarray(pygeoda.redcap(K, w, df, 'fullorder-averagelinkage')['Clusters'])
        lab['schc_ward'] = np.asarray(pygeoda.schc(K, w, df, 'ward')['Clusters'])
    return lab


def verdad_simulaciones():
    """RBTSS of the true partition, of SDEC and of the other methods on the 100 draws of the Monte Carlo comparison
    (labels read from resultados/montecarlo_a<amplitude>.jsonl). Shows whether a higher RBTSS on the original
    variables indicates a better recovery of the true regions."""
    import json
    from dgp_simulacion import genera_datos
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    filas = []
    for a in (0.5, 1.0):
        ruta = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_a%g.jsonl' % a)
        if not os.path.exists(ruta):
            continue
        for r in (json.loads(l) for l in open(ruta)):
            X = genera_datos(obj, a, r['semilla_datos'])
            fila = {'amplitud': a, 'replica': r['replica'], 'verdad': metricas.rbtss(X, obj)}
            for m in ('sdec', 'sh', 'pca6_ward', 'skater', 'redcap', 'schc_ward', 'suav2'):
                if m in r['grupos']:
                    fila[m] = metricas.rbtss(X, np.asarray(r['grupos'][m]))
            filas.append(fila)
    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'rbtss_verdad_simulaciones.csv'), index=False)
    if len(out):
        res = out.drop(columns='replica').groupby('amplitud').mean()
        for m in res.columns.drop('verdad'):
            res['%s_sobre_verdad' % m] = out.groupby('amplitud').apply(lambda d: (d[m] > d['verdad']).sum())
        print(res.round(3).T.to_string())


def main():
    verdad_simulaciones()
    filas = []
    for nombre, cargar, K, k in (('covid', datos.datos_covid, constantes.K_COVID, constantes.KNN_COVID),
                                 ('salarios', datos.datos_salarios, constantes.K_SAL, constantes.KNN_SAL)):
        d = cargar()
        lab = particiones(d, K, k)
        for m, g in lab.items():
            g = np.asarray(g)
            tam = np.bincount(pd.factorize(g)[0])
            filas.append({'aplicacion': nombre, 'metodo': m, 'K': K, 'k': k,
                          'rbtss_original': metricas.rbtss(d['prop_st'], g),
                          'grupos': int(len(tam)), 'menor': int(tam.min()), 'mayor': int(tam.max()),
                          'unitarios': int((tam == 1).sum())})
        filas.append({'aplicacion': nombre, 'metodo': 'aleatorio', 'K': K, 'k': k,
                      'rbtss_original': metricas.piso_aleatorio(d['prop_st'], lab['sh'])})
    out = pd.DataFrame(filas)
    ruta = os.path.join(constantes.DIR_RESULTADOS, 'rbtss_comparadores.csv')
    out.to_csv(ruta, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
