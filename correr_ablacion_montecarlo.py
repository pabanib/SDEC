# -*- coding: utf-8 -*-
"""
Key variants of the ablation over draws of the data (the draws of the Monte Carlo comparison, correr_montecarlo.py).

Usage:  python correr_ablacion_montecarlo.py <amplitude> [replications] [first replication]
        python correr_ablacion_montecarlo.py 0.5 30
        python correr_ablacion_montecarlo.py 1.0 30

In replication r the data are those of correr_montecarlo.py (seed SEMILLA_BASE + r), K = 6 and the same 6-nearest-
neighbor matrix. The autoencoder is pre-trained once with seed r and every variant starts from those same weights and
re-seeds keras with r (experimentos.refinar), so the variants differ only in the component that is modified:
  sdec          (i)   spatial target (neighbors only) + constrained Ward               [the method]
  argmax        (ii)  spatial target + argmax of Q
  propio        (new) target with the neighbors and the unit's own DEC term (W + I) + constrained Ward
  xie_ward      (iii) target of Xie et al. (2016) + constrained Ward
  dec           (new) target of Xie et al. + argmax of Q, that is, DEC as published
  sin_refinar   (iv)  constrained Ward on the pre-trained embedding, without refinement
For sdec it also stores the KL trace of the refinement (deep_cluster.sdec.traza) and the adjusted Rand index between
the final partition and the argmax of Q of the same trained network.
One row per replication in resultados/ablacion_montecarlo_a<amplitude>.jsonl; replications already there are skipped.
Each replication runs five refinements (about 15 to 20 minutes on a laptop CPU); the two amplitudes write to different
files and can run at the same time.
"""
import sys
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
import constantes
from sdec import datos
from sdec.experimentos import entrenar, refinar, leer_jsonl, agregar_jsonl
from dgp_simulacion import genera_datos
from correr_montecarlo import SEMILLA_BASE

VARIANTES = (('sdec', 'espacial', 'ward'), ('argmax', 'espacial', 'argmax'), ('propio', 'espacial_propio', 'ward'),
             ('xie_ward', 'xie', 'ward'), ('dec', 'xie', 'argmax'))


def main():
    a = float(sys.argv[1])
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    primera = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    K, k = constantes.K_SIM, constantes.KNN_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    archivo = os.path.join(constantes.DIR_RESULTADOS, 'ablacion_montecarlo_a%g.jsonl' % a)
    hechas = {f['replica'] for f in leer_jsonl(archivo)}
    for r in range(primera, primera + R):
        if r in hechas:
            continue
        X = genera_datos(obj, a, SEMILLA_BASE + r)
        T = X.shape[1] // 2
        sc, w0 = entrenar(X, K, T, r)
        sc.model.set_weights(w0)
        emb = sc.encoder.predict([X[:, :T], X[:, T:]], verbose=0)
        lab = {'sin_refinar': AgglomerativeClustering(K, connectivity=W, linkage='ward').fit_predict(emb)}
        fila = {'amplitud': a, 'replica': r, 'semilla_datos': SEMILLA_BASE + r, 'iteraciones': {}}
        for nombre, objetivo, final in VARIANTES:
            g, it = refinar(sc, X, W, T, objetivo, final, r, w0)
            lab[nombre] = g
            fila['iteraciones'][nombre] = int(it)
            if nombre == 'sdec':
                q = sc.model.predict([X[:, :T], X[:, T:]], verbose=0)
                fila['ari_final_argmax'] = float(ari(g, q.argmax(1)))
                fila['traza_sdec'] = [list(t) for t in sc.traza]
        fila['grupos'] = {m: [int(v) for v in np.asarray(g)] for m, g in lab.items()}
        agregar_jsonl(archivo, fila)
        print('a', a, 'replication', r, 'AMI', {m: round(float(ami(obj, np.asarray(g))), 3) for m, g in lab.items()},
              flush=True)


if __name__ == '__main__':
    main()
