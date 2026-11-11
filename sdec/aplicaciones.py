# -*- coding: utf-8 -*-
"""Runs of SDEC on the two applications (COVID-19 and wages and jobs): training, per-run summaries, summary across
seeds and contiguity of the regions between polygons."""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import geopandas as gpd
import constantes
from sdec import datos, metricas
from sdec.experimentos import fijar_semilla, leer_jsonl, agregar_jsonl
from sdec import deep_cluster as dc

CONFIG = {
    'covid': {'K': constantes.K_COVID, 'T': constantes.T_COVID, 'k': constantes.KNN_COVID, 'cargar': datos.datos_covid},
    'salarios': {'K': constantes.K_SAL, 'T': constantes.T_SAL, 'k': constantes.KNN_SAL, 'cargar': datos.datos_salarios},
}


def correr(prop_st, W, K, T, semilla):
    """One run of SDEC: the autoencoder reconstructs the standardized matrix and the partition is the final one."""
    fijar_semilla(semilla)
    sc = dc.sdec(K, [T, T])
    sc.gen_modelo()
    X = [prop_st[:, :T], prop_st[:, T:]]
    sc.train_autoencoder(X, prop_st)
    sc.ajustar_modelo(X, W, semilla=semilla)
    return sc


def matriz_contiguidad(geo, tol=2000.0):
    """Polygon contiguity: each geometry is widened by `tol` meters and two departments are neighbors if the widened
    geometries intersect, that is, if their boundaries are less than 2 * tol (4 km) apart, a lenient tolerance."""
    g = gpd.GeoDataFrame(geometry=geo.values.buffer(tol))
    idx = g.sindex
    n = len(g)
    filas, cols = [], []
    for i, geom in enumerate(g.geometry):
        for j in idx.query(geom, predicate='intersects'):
            if j > i:
                filas += [i, j]
                cols += [j, i]
    return sp.csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(n, n))


def contiguidad_grupos(A, grupos):
    """Number of regions that are not connected in the contiguity graph and number of departments that are
    separated from the main component of their region."""
    no_conexas, fuera = 0, 0
    for k in np.unique(grupos):
        m = np.where(grupos == k)[0]
        nc, lab = connected_components(A[m][:, m])
        if nc > 1:
            no_conexas += 1
            tam = np.bincount(lab)
            fuera += int(tam.sum() - tam.max())
    return int(len(np.unique(grupos))), no_conexas, fuera


def archivo(nombre):
    return os.path.join(constantes.DIR_RESULTADOS, '%s_semillas.jsonl' % nombre)


def semillas(nombre, n):
    """Runs the seeds 1..n that are not yet in resultados/<nombre>_semillas.jsonl (one row per seed)."""
    c = CONFIG[nombre]
    d = c['cargar']()
    hechas = {f['semilla'] for f in leer_jsonl(archivo(nombre))}
    for s in range(1, n + 1):
        if s in hechas:
            continue
        sc = correr(d['prop_st'], d['W'], c['K'], c['T'], s)
        fila = {'semilla': s, 'resumen': metricas.resumen_corrida(sc, d['prop_st'], c['T']),
                'grupos': [int(x) for x in sc.y_pred]}
        agregar_jsonl(archivo(nombre), fila)
        r = fila['resumen']
        print('%s seed %d | groups %d | RBTSS original %.3f, encoder %.3f | last update %s' %
              (nombre, s, r['n_grupos'], r['rbtss_original'], r['rbtss_encoder'], r['ultima_actualizacion']), flush=True)


def _ms(v):
    v = np.array(v, float)
    return {'media': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None,
            'min': float(v.min()), 'max': float(v.max())}


def resumen(nombre):
    """Summary across seeds: mean, standard deviation, minimum and maximum of the metrics, sizes of the smallest and
    largest region, ARI between runs and contiguity between polygons of the regions."""
    c = CONFIG[nombre]
    d = c['cargar']()
    filas = leer_jsonl(archivo(nombre))
    out = {'n_corridas': len(filas)}
    for k in ('coseno_medio', 'rbtss_original', 'rbtss_encoder', 'piso_aleatorio_original', 'ultima_actualizacion'):
        out[k] = _ms([f['resumen'][k] for f in filas])
    for m in ('silueta', 'calinski_harabasz', 'davies_bouldin'):
        out['encoder.' + m] = _ms([f['resumen']['intrinsecas_encoder'][m] for f in filas])
    tam = [f['resumen']['tamanos'] for f in filas]
    out['region_mas_chica'] = {'min': int(min(t[0] for t in tam)), 'max': int(max(t[0] for t in tam))}
    out['region_mas_grande'] = {'min': int(min(t[-1] for t in tam)), 'max': int(max(t[-1] for t in tam))}
    g = [np.asarray(f['grupos']) for f in filas]
    pares = [ari(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]
    if pares:
        out['ari_entre_semillas'] = {'media': float(np.mean(pares)), 'min': float(np.min(pares)),
                                     'max': float(np.max(pares))}
    out['corridas_con_parada_antes_del_maximo'] = int(sum(1 for f in filas if f['resumen']['ultima_actualizacion'] < 980))
    A = matriz_contiguidad(d['geo'])
    cont = pd.DataFrame([dict(zip(('regiones', 'no_conexas', 'deptos_fuera'), contiguidad_grupos(A, x))) for x in g])
    cont.insert(0, 'semilla', [f['semilla'] for f in filas])
    out['contiguidad_poligonos_4km'] = {
        'no_conexas': {'media': float(cont.no_conexas.mean()), 'min': int(cont.no_conexas.min()),
                       'max': int(cont.no_conexas.max())},
        'deptos_fuera': {'media': float(cont.deptos_fuera.mean()), 'min': int(cont.deptos_fuera.min()),
                         'max': int(cont.deptos_fuera.max())},
        'corridas_con_alguna_region_no_conexa': int((cont.no_conexas > 0).sum())}
    cont.to_csv(os.path.join(constantes.DIR_RESULTADOS, '%s_contiguidad_por_corrida.csv' % nombre), index=False)
    ruta = os.path.join(constantes.DIR_RESULTADOS, '%s_resumen.json' % nombre)
    json.dump(out, open(ruta, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return out


def repetir(nombre, semilla, veces=2):
    """Runs the same seed several times in the same process and compares the partitions."""
    c = CONFIG[nombre]
    d = c['cargar']()
    etiquetas = [np.asarray(correr(d['prop_st'], d['W'], c['K'], c['T'], semilla).y_pred) for _ in range(veces)]
    iguales = all(np.array_equal(etiquetas[0], e) for e in etiquetas[1:])
    aris = [float(ari(etiquetas[0], e)) for e in etiquetas[1:]]
    print('%s, seed %d, %d repetitions: identical partitions = %s, ARI between repetitions = %s' %
          (nombre, semilla, veces, iguales, aris))
    json.dump({'semilla': semilla, 'repeticiones': veces, 'particiones_identicas': bool(iguales),
               'ari_entre_repeticiones': aris},
              open(os.path.join(constantes.DIR_RESULTADOS, '%s_repeticion.json' % nombre), 'w'), indent=1)


def main(nombre, argv):
    """python correr_<nombre>.py semillas [n]  |  resumen  |  repetir [semilla]"""
    modo = argv[0] if argv else 'semillas'
    if modo == 'semillas':
        semillas(nombre, int(argv[1]) if len(argv) > 1 else 30)
        resumen(nombre)
    elif modo == 'resumen':
        resumen(nombre)
    elif modo == 'repetir':
        repetir(nombre, int(argv[1]) if len(argv) > 1 else 42)
    else:
        raise SystemExit(main.__doc__)
