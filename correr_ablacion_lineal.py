# -*- coding: utf-8 -*-
"""
Ablation of the nonlinearity of the encoder (Table 5, variants (xi) and (xii)).

Usage:  python correr_ablacion_lineal.py <scenario 1|2> [seed ...]      (default seeds: 1 to 5)

For each seed two autoencoders are trained with the same architecture, seed, optimizer and loss: the network of SDEC
(ReLU activations) and the same network with the identity as activation in every layer. From the pre-trained weights of
each one:
  sdec_no_lineal    : complete SDEC with the nonlinear network
  sdec_lineal       : complete SDEC with the linear network                                                  (xi)
  ae_no_lineal_ward : constrained Ward on the nonlinear embedding, without refinement
  ae_lineal_ward    : constrained Ward on the linear embedding, without refinement                           (xii)
All variants run in the same environment and with the same seeds, so that the comparison between them does not depend
on the software version or the hardware. Appends one row per variant and seed (labels, AMI and ARI with the target
partition) to resultados/ablacion_lineal_escenario<scenario>.jsonl.
"""
import sys
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
import constantes
from sdec import datos
from sdec.experimentos import entrenar, refinar, leer_jsonl, agregar_jsonl
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
    arch = os.path.join(constantes.DIR_RESULTADOS, 'ablacion_lineal_escenario%d.jsonl' % esc)
    hechas = {(f['variante'], f['semilla']) for f in leer_jsonl(arch)}

    def guardar(variante, semilla, lab, extra=None):
        fila = {'escenario': esc, 'variante': variante, 'semilla': semilla, 'grupos': [int(v) for v in lab],
                'AMI': float(ami(obj, lab)), 'ARI': float(ari(obj, lab))}
        if extra:
            fila.update(extra)
        agregar_jsonl(arch, fila)
        hechas.add((variante, semilla))
        print('scenario', esc, 'seed', semilla, variante, 'AMI %.3f' % fila['AMI'], extra or '', flush=True)

    for s in semillas:
        for etiqueta, activacion in (('no_lineal', 'relu'), ('lineal', 'linear')):
            if ('sdec_' + etiqueta, s) in hechas and ('ae_%s_ward' % etiqueta, s) in hechas:
                continue
            sc, w0 = entrenar(X, K, T, s, activacion=activacion)
            emb = sc.encoder.predict([X[:, :T], X[:, T:]], verbose=0)
            if ('ae_%s_ward' % etiqueta, s) not in hechas:
                guardar('ae_%s_ward' % etiqueta, s, AgglomerativeClustering(K, connectivity=W).fit(emb).labels_)
            if ('sdec_' + etiqueta, s) not in hechas:
                lab, it = refinar(sc, X, W, T, 'espacial', 'ward', s, w0)
                guardar('sdec_' + etiqueta, s, lab, {'iteraciones': int(it)})


if __name__ == '__main__':
    main()
