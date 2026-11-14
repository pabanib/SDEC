# -*- coding: utf-8 -*-
"""
Ablations of SDEC and temporal permutation (Table 5, variants (i) to (ix)).

Usage:  python correr_ablaciones.py <scenario 1|2> [seed ...]      (default seeds: 1 to 5)

For each seed ONE autoencoder is trained and, from those same weights, the following variants are obtained:
  sdec_completo : refinement with the spatially smoothed target + constrained Ward on Q (SDEC)               (i)
  sdec_sin_ward : spatial target, final partition by the argmax of Q                                          (ii)
  dec_xie_ward  : target of Xie et al. (2016) (no neighbor term) + constrained Ward on Q                      (iii)
  ae_ward       : constrained Ward on the autoencoder embedding, without refinement                           (iv)
  ae_kmeans     : k-means on the same embedding                                                               (v)
  permutado     : complete SDEC with the T periods permuted (the same permutation for all units and both
                  variables), trained from scratch with the same seed                                          (ix)
Deterministic variants (one run per scenario, seed 0):
  pca6_ward, pca2_ward : constrained Ward on the first 6 or 2 principal components                              (vi), (vii)
  sh                   : constrained Ward on the original variables                                           (viii)
Appends one row per variant and seed (with the labels) to resultados/ablaciones_escenario<scenario>.jsonl.
Each seed takes about 5 minutes on a CPU.
"""
import sys
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import constantes
from sdec import datos
from sdec.experimentos import entrenar, refinar, fijar_semilla, leer_jsonl, agregar_jsonl
from dgp_simulacion import escenario


def main():
    esc = int(sys.argv[1])
    semillas = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4, 5]
    K = constantes.K_SIM
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    X = escenario(obj, esc, constantes.SEMILLA_DATOS)
    T = X.shape[1] // 2
    arch = os.path.join(constantes.DIR_RESULTADOS, 'ablaciones_escenario%d.jsonl' % esc)
    hechas = {(f['variante'], f['semilla']) for f in leer_jsonl(arch)}

    def guardar(variante, semilla, lab, extra=None):
        fila = {'escenario': esc, 'variante': variante, 'semilla': semilla, 'grupos': [int(v) for v in lab]}
        if extra:
            fila.update(extra)
        agregar_jsonl(arch, fila)
        hechas.add((variante, semilla))

    if ('sh', 0) not in hechas:
        guardar('sh', 0, AgglomerativeClustering(K, connectivity=W).fit(X).labels_)
    for h in (6, 2):
        if ('pca%d_ward' % h, 0) not in hechas:
            Z = PCA(h).fit_transform(X)
            guardar('pca%d_ward' % h, 0, AgglomerativeClustering(K, connectivity=W).fit(Z).labels_)

    perm = np.random.RandomState(123).permutation(T)
    Xp = np.c_[X[:, :T][:, perm], X[:, T:][:, perm]]
    de_la_red = ('sdec_completo', 'sdec_sin_ward', 'dec_xie_ward', 'ae_ward', 'ae_kmeans')
    for s in semillas:
        if not all((v, s) in hechas for v in de_la_red):
            sc, w0 = entrenar(X, K, T, s)
            emb = sc.encoder.predict([X[:, :T], X[:, T:]], verbose=0)
            if ('ae_ward', s) not in hechas:
                guardar('ae_ward', s, AgglomerativeClustering(K, connectivity=W).fit(emb).labels_)
            if ('ae_kmeans', s) not in hechas:
                guardar('ae_kmeans', s, KMeans(K, random_state=s, n_init=10).fit(emb).labels_)
            for var, objetivo, final in [('sdec_completo', 'espacial', 'ward'), ('sdec_sin_ward', 'espacial', 'argmax'),
                                         ('dec_xie_ward', 'xie', 'ward')]:
                if (var, s) in hechas:
                    continue
                lab, it = refinar(sc, X, W, T, objetivo, final, s, w0)
                guardar(var, s, lab, {'iteraciones': int(it)})
                print('scenario', esc, 'seed', s, var, 'groups', len(np.unique(lab)), 'it', it, flush=True)
        if ('permutado', s) not in hechas:
            sp, w0p = entrenar(Xp, K, T, s)
            lab, it = refinar(sp, Xp, W, T, 'espacial', 'ward', s, w0p)
            guardar('permutado', s, lab, {'iteraciones': int(it)})
            print('scenario', esc, 'seed', s, 'permutado', 'groups', len(np.unique(lab)), 'it', it, flush=True)


if __name__ == '__main__':
    main()
