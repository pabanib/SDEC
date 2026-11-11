# -*- coding: utf-8 -*-
"""
Extrinsic metrics of the two scenarios (adjusted mutual information, mutual information, homogeneity and Rand score
against the target partition): SDEC (mean and standard deviation over the seeds of resultados/simulaciones_sdec.jsonl),
SH (constrained Ward on the data, computed here) and, if resultados/comparadores_escenario<k>.csv exist (comparadores.R),
SKATER, REDCAP and SCHC. Also summarizes the sensitivity to k (resultados/sensibilidad_k.jsonl) if it exists.
Usage:  python tablas_simulaciones.py        Writes resultados/tabla_metricas_extrinsecas.csv and prints the tables.
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_mutual_info_score as ami, mutual_info_score as mi, homogeneity_score as hom,
                             rand_score)
import constantes
from sdec import datos
from sdec.experimentos import leer_jsonl
from dgp_simulacion import escenario

METRICAS = [('AMI', ami), ('MI', mi), ('Homogeneity', hom), ('Rand', rand_score)]


def main():
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    filas = leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'simulaciones_sdec.jsonl'))
    out = []
    for esc in (1, 2):
        X = escenario(obj, esc, constantes.SEMILLA_DATOS)
        met = {'SH': AgglomerativeClustering(constantes.K_SIM, connectivity=W).fit(X).labels_}
        ruta = os.path.join(constantes.DIR_RESULTADOS, 'comparadores_escenario%d.csv' % esc)
        if os.path.exists(ruta):
            c = pd.read_csv(ruta)
            for m in ('skater', 'redcap', 'schc'):
                met[m.upper()] = c[m].values
        sd = [np.asarray(f['grupos']) for f in filas if f['escenario'] == esc]
        for nm, f in METRICAS:
            fila = {'escenario': esc, 'metrica': nm}
            if sd:
                v = np.array([f(obj, g) for g in sd])
                fila['SDEC_media'], fila['SDEC_sd'], fila['SDEC_n'] = v.mean(), v.std(ddof=1), len(v)
            for m, lab in met.items():
                fila[m] = f(obj, lab)
            out.append(fila)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'tabla_metricas_extrinsecas.csv'), index=False)
    print(t.round(4).to_string(index=False))
    sk = leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'sensibilidad_k.jsonl'))
    if sk:
        d = pd.DataFrame([{'escenario': f['escenario'], 'k': f['k'], 'ami': f['ami']} for f in sk])
        r = d.groupby(['escenario', 'k'])['ami'].agg(['mean', 'std', 'count'])
        r.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'sensibilidad_k_resumen.csv'))
        print('\nSensitivity to k (AMI):\n', r.round(3).to_string())


if __name__ == '__main__':
    main()
