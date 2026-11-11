# -*- coding: utf-8 -*-
"""
Control without neural network (Table 4, variant (x)): the data are smoothed m times with the row-standardized neighbor
matrix (each unit is replaced by the average of its neighbors) and constrained Ward is applied, with and without a
previous PCA(6). AMI with the target partition in the two scenarios.
Usage:  python control_suavizado.py        Writes resultados/control_suavizado.json
"""
import os
import json
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos
from dgp_simulacion import escenario


def main():
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    Ws = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    Wr = Ws.multiply(1 / Ws.sum(1)).tocsr()
    K = constantes.K_SIM
    res = {}
    for esc in (1, 2):
        X = escenario(obj, esc, constantes.SEMILLA_DATOS)
        r = {'SH (m=0)': ami(obj, AC(K, connectivity=Ws).fit(X).labels_)}
        Z = X.copy()
        for m in (1, 2, 3):
            Z = Wr @ Z
            r['Ward on W^%d X' % m] = ami(obj, AC(K, connectivity=Ws).fit(Z).labels_)
            r['PCA(6) on W^%d X + Ward' % m] = ami(obj, AC(K, connectivity=Ws).fit(PCA(6).fit_transform(Z)).labels_)
        res['escenario%d' % esc] = r
        print('scenario', esc, {k: round(v, 3) for k, v in r.items()})
    json.dump(res, open(os.path.join(constantes.DIR_RESULTADOS, 'control_suavizado.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
