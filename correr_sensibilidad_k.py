# -*- coding: utf-8 -*-
"""
Sensitivity of SDEC to the number of neighbors k of the spatial weights matrix (k = 4, 5, 6) in the two scenarios.

Usage:  python correr_sensibilidad_k.py <scenario 1|2> <k> [seed ...]      (default seeds: 1 to 10)
Appends one row per run to resultados/sensibilidad_k.jsonl.
"""
import sys
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos
from sdec.experimentos import entrenar, leer_jsonl, agregar_jsonl
from dgp_simulacion import escenario

ARCHIVO = os.path.join(constantes.DIR_RESULTADOS, 'sensibilidad_k.jsonl')


def main():
    esc, k = int(sys.argv[1]), int(sys.argv[2])
    semillas = [int(s) for s in sys.argv[3:]] or list(range(1, 11))
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, k)
    X = escenario(obj, esc, constantes.SEMILLA_DATOS)
    T = X.shape[1] // 2
    hechas = {(f['escenario'], f['k'], f['semilla']) for f in leer_jsonl(ARCHIVO)}
    for s in semillas:
        if (esc, k, s) in hechas:
            continue
        sc, w0 = entrenar(X, constantes.K_SIM, T, s)
        sc.ajustar_modelo([X[:, :T], X[:, T:]], W, semilla=s)
        lab = np.asarray(sc.y_pred)
        agregar_jsonl(ARCHIVO, {'escenario': esc, 'k': k, 'semilla': s, 'ami': float(ami(obj, lab)),
                                'grupos': [int(v) for v in lab]})
        print('scenario', esc, 'k', k, 'seed', s, 'AMI %.4f' % ami(obj, lab), flush=True)


if __name__ == '__main__':
    main()
