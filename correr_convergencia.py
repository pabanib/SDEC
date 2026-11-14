# -*- coding: utf-8 -*-
"""
Convergence of the refinement in the two applications: KL trace and partition after the usual maximum of 1,000
iterations and after a longer run.

Usage:  python correr_convergencia.py <covid or salarios> [seeds] [maximum iterations]
        python correr_convergencia.py covid 5 5000
        python correr_convergencia.py salarios 5 5000

For each seed s = 1..n the run is exactly that of correr_covid.py / correr_salarios.py (same data, K, k and seed),
except that the maximum number of iterations is raised. Stored per seed, in resultados/convergencia_<case>.jsonl:
  traza            at each update (every 20 iterations): iteration, KL(P_prev||Q) (the loss that the last 20 iterations
                   minimized, evaluated on all units), KL(P_new||Q), the proportion of units whose label changes (the
                   quantity used by the stopping rule, which depends on the numbering of the groups returned by the
                   constrained Ward step) and the adjusted Rand index with the previous partition (which does not)
  particion_980    partition at iteration 980, the one reported in the paper when the stopping rule is not met
  particion_final  partition at the end of the long run (stopping rule met or maximum reached)
  ari_980_final    adjusted Rand index between the two
  rbtss            RBTSS on the original variables of both partitions
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sdec import metricas
from sdec.aplicaciones import CONFIG
from sdec.experimentos import fijar_semilla, leer_jsonl, agregar_jsonl
from sdec import deep_cluster as dc
import constantes


def main():
    caso = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    c = CONFIG[caso]
    d = c['cargar']()
    X = [d['prop_st'][:, :c['T']], d['prop_st'][:, c['T']:]]
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'convergencia_%s.jsonl' % caso)
    hechas = {f['semilla'] for f in leer_jsonl(archivo)}
    for s in range(1, n + 1):
        if s in hechas:
            continue
        fijar_semilla(s)
        sc = dc.sdec(c['K'], [c['T'], c['T']])
        sc.gen_modelo()
        sc.train_autoencoder(X, d['prop_st'])
        sc.ajustar_modelo(X, d['W'], semilla=s, max_iter=max_iter, guardar=(980,))
        g980 = sc.particiones.get(980)
        gfin = np.asarray(sc.y_pred)
        fila = {'caso': caso, 'semilla': s, 'max_iter': max_iter, 'n_iter': int(sc.n_iter),
                'traza': [list(t) for t in sc.traza],
                'particion_980': None if g980 is None else [int(v) for v in g980],
                'particion_final': [int(v) for v in gfin],
                'ari_980_final': None if g980 is None else float(ari(g980, gfin)),
                'rbtss': {'980': None if g980 is None else metricas.rbtss(d['prop_st'], g980),
                          'final': metricas.rbtss(d['prop_st'], gfin)}}
        agregar_jsonl(archivo, fila)
        print(caso, 'seed', s, 'iterations', fila['n_iter'], 'ARI 980 vs final', fila['ari_980_final'],
              'last change', sc.traza[-1][3], flush=True)


if __name__ == '__main__':
    main()
