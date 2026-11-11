# -*- coding: utf-8 -*-
"""
SDEC on the two simulated scenarios (Section 4), one run per seed.

Usage:  python correr_simulaciones.py <scenario 1|2> [seed ...]      (default seeds: 1 to 10)

For each seed a new autoencoder is trained (100 epochs) and refined; the partition (labels of the 525 departments)
is appended to resultados/simulaciones_sdec.jsonl as {'escenario', 'semilla', 'grupos', 'ami', 'rand'}. Seeds already
in the file are skipped. Each run takes 1 to 3 minutes on a CPU.
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami, rand_score
import constantes
from sdec import datos
from sdec.experimentos import entrenar, leer_jsonl, agregar_jsonl
from dgp_simulacion import escenario

ARCHIVO = os.path.join(constantes.DIR_RESULTADOS, 'simulaciones_sdec.jsonl')


def main():
    esc = int(sys.argv[1])
    semillas = [int(s) for s in sys.argv[2:]] or list(range(1, 11))
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    X = escenario(obj, esc, constantes.SEMILLA_DATOS)
    T = X.shape[1] // 2
    hechas = {(f['escenario'], f['semilla']) for f in leer_jsonl(ARCHIVO)}
    for s in semillas:
        if (esc, s) in hechas:
            continue
        sc, w0 = entrenar(X, constantes.K_SIM, T, s)
        sc.ajustar_modelo([X[:, :T], X[:, T:]], W, semilla=s)
        lab = np.asarray(sc.y_pred)
        fila = {'escenario': esc, 'semilla': s, 'grupos': [int(v) for v in lab],
                'ami': float(ami(obj, lab)), 'rand': float(rand_score(obj, lab))}
        agregar_jsonl(ARCHIVO, fila)
        print('scenario', esc, 'seed', s, 'AMI %.4f' % fila['ami'], 'Rand %.4f' % fila['rand'],
              'groups', len(np.unique(lab)), flush=True)


if __name__ == '__main__':
    main()
