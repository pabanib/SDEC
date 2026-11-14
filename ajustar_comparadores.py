# -*- coding: utf-8 -*-
"""
Tuning budget for the deep comparison methods (SDCN, graph autoencoder, Region2Vec), to remove the advantage of the
designer: SDEC was configured with preliminary tests on this data generating process, whereas the comparison methods
were run with the settings of their reference implementations.

Usage:
  python ajustar_comparadores.py ajuste <amplitude>      grid of configurations on 10 tuning draws (replications
                                                          1001 to 1010, not used in the Monte Carlo comparison)
  python ajustar_comparadores.py evaluar <amplitude>     for each output of each method, the configuration with the
                                                          highest mean AMI on the tuning draws is run on replications
                                                          1 to 100 (the draws of Table 2)

The configuration is chosen with the AMI against the true partition, which SDEC's preliminary tests also used; since
the tuning draws are independent of the evaluation draws, the selection does not use the data on which the methods
are compared, and it is favorable to the comparison methods. Outputs:
  resultados/ajuste_comparadores_a<amplitude>.jsonl     one row per (method, configuration, tuning draw)
  resultados/montecarlo_ajustados_a<amplitude>.jsonl    one row per evaluation draw, labels of every tuned output
Both are resumable.
"""
import sys
import os
import json
import itertools
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos, profundos
from sdec.region2vec import region2vec
from sdec.experimentos import leer_jsonl, agregar_jsonl
from dgp_simulacion import genera_datos
from correr_montecarlo import SEMILLA_BASE

AJUSTE = range(1001, 1011)

REJILLAS = {
    'sdcn': [dict(capas=c, lr=lr, epocas_pre=pe, epocas=ep)
             for c, lr, pe, ep in itertools.product([(500, 500, 2000), (256, 128), (128, 64, 32)], [1e-3, 1e-4],
                                                    [30, 100], [200])],
    'gae': [dict(ocultas=o, lr=lr, epocas=ep)
            for o, lr, ep in itertools.product([(32, 16), (64, 32), (128, 64), (32, 6)], [0.01, 0.001], [200, 500])],
    'region2vec': [dict(ocultas=o, salida=s, lr=lr, saltos=h)
                   for (o, s), lr, h in itertools.product([(16, 14), (64, 32), (32, 6)], [1e-3, 1e-2], [3, 5])],
}
FUNCIONES = {'sdcn': profundos.sdcn, 'gae': profundos.gae, 'region2vec': region2vec}


def correr(metodo, cfg, X, W, K, semilla):
    out = FUNCIONES[metodo](X, W, K, semilla, **cfg)
    return {m: np.asarray(g) for m, g in out.items() if not m.endswith('_epocas')}


def ajuste(a, obj, W, K):
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'ajuste_comparadores_a%g.jsonl' % a)
    hechas = {(f['metodo'], f['config'], f['replica']) for f in leer_jsonl(archivo)}
    for r in AJUSTE:
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        for metodo, rejilla in REJILLAS.items():
            for i, cfg in enumerate(rejilla):
                if (metodo, i, r) in hechas:
                    continue
                lab = correr(metodo, cfg, X, W, K, r)
                agregar_jsonl(archivo, {'amplitud': a, 'replica': r, 'metodo': metodo, 'config': i,
                                        'parametros': json.loads(json.dumps(cfg)),
                                        'ami': {m: float(ami(obj, g)) for m, g in lab.items()}})
                print('tuning a', a, 'draw', r, metodo, i, {m: round(float(ami(obj, g)), 3) for m, g in lab.items()},
                      flush=True)


def mejores(a):
    """For each output, the configuration with the highest mean AMI over the tuning draws."""
    filas = leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'ajuste_comparadores_a%g.jsonl' % a))
    medias = {}
    for f in filas:
        for salida, v in f['ami'].items():
            medias.setdefault((f['metodo'], salida, f['config']), []).append(v)
    sel = {}
    for (metodo, salida, i), v in medias.items():
        if len(v) < len(AJUSTE):
            continue
        if salida not in sel or np.mean(v) > sel[salida][2]:
            sel[salida] = (metodo, i, float(np.mean(v)))
    return sel


def evaluar(a, obj, W, K):
    sel = mejores(a)
    print('selected configurations (method, configuration, mean AMI on the tuning draws):', sel, flush=True)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'montecarlo_ajustados_a%g.jsonl' % a)
    hechas = {f['replica'] for f in leer_jsonl(archivo)}
    configs = sorted({(m, i) for m, i, _ in sel.values()})
    for r in range(1, 101):
        if r in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        lab = {}
        for metodo, i in configs:
            out = correr(metodo, REJILLAS[metodo][i], X, W, K, r)
            for salida, g in out.items():
                if sel.get(salida, (None, None))[:2] == (metodo, i):
                    lab[salida + '_ajustado'] = g
        agregar_jsonl(archivo, {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r,
                                'seleccion': {s: [m, i] for s, (m, i, _) in sel.items()},
                                'grupos': {m: [int(v) for v in g] for m, g in lab.items()}})
        print('evaluation a', a, 'draw', r, {m: round(float(ami(obj, g)), 3) for m, g in lab.items()}, flush=True)


def main():
    modo, a = sys.argv[1], float(sys.argv[2])
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    (ajuste if modo == 'ajuste' else evaluar)(a, obj, W, K)


if __name__ == '__main__':
    main()
