# -*- coding: utf-8 -*-
"""
Summary of the ablations (Table 4): AMI and ARI with the target partition (mean and standard deviation over seeds) of
each variant and scenario, and the temporal permutation test: ARI between the permuted SDEC and the complete SDEC with the
same seed, against the ARI between complete runs with different seeds.
Reads resultados/ablaciones_escenario<k>.jsonl, resultados/ablacion_lineal_escenario<k>.jsonl and
resultados/control_suavizado.json. Usage:  python analizar_ablaciones.py     Writes resultados/ablaciones_resumen.json
"""
import os
import json
import itertools
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
import constantes
from sdec import datos
from sdec.experimentos import leer_jsonl

ORDEN = ['sdec_completo', 'sdec_sin_ward', 'dec_xie_ward', 'ae_ward', 'ae_kmeans', 'permutado', 'pca6_ward', 'pca2_ward',
         'sh']


def ms(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float('nan'), len(v))


def agrupar(filas):
    por = {}
    for r in filas:
        por.setdefault(r['variante'], {})[r['semilla']] = np.array(r['grupos'])
    return por


def main():
    obj = datos.objetivo_simulacion(datos.departamentos())
    res = {}
    for esc in (1, 2):
        por = agrupar(leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'ablaciones_escenario%d.jsonl' % esc)))
        lin = agrupar(leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'ablacion_lineal_escenario%d.jsonl' % esc)))
        por.update({k: v for k, v in lin.items() if k in ('sdec_lineal', 'ae_lineal_ward', 'sdec_no_lineal',
                                                          'ae_no_lineal_ward')})
        out = {'variantes': {}}
        for v in ORDEN + ['sdec_lineal', 'ae_lineal_ward', 'sdec_no_lineal', 'ae_no_lineal_ward']:
            if v in por:
                out['variantes'][v] = {'AMI': ms([ami(obj, l) for l in por[v].values()]),
                                       'ARI': ms([ari(obj, l) for l in por[v].values()])}
        full = por.get('sdec_completo', {})
        if len(full) > 1:
            out['ARI_entre_semillas_sdec_completo'] = ms([ari(full[a], full[b])
                                                          for a, b in itertools.combinations(sorted(full), 2)])
        for nm, v in (('ARI_permutado_vs_completo_misma_semilla', 'permutado'),
                      ('ARI_argmax_vs_ward_misma_semilla', 'sdec_sin_ward')):
            if full and v in por:
                comunes = sorted(set(full) & set(por[v]))
                out[nm] = ms([ari(full[s], por[v][s]) for s in comunes])
        res['escenario%d' % esc] = out
    ruta = os.path.join(constantes.DIR_RESULTADOS, 'control_suavizado.json')
    if os.path.exists(ruta):
        res['control_suavizado'] = json.load(open(ruta))
    json.dump(res, open(os.path.join(constantes.DIR_RESULTADOS, 'ablaciones_resumen.json'), 'w'), indent=1)
    for esc in (1, 2):
        o = res['escenario%d' % esc]
        print('\n== scenario', esc)
        for v, d in o['variantes'].items():
            print('%-20s AMI %.3f (%.3f) n=%d | ARI %.3f (%.3f)' % (v, d['AMI'][0], d['AMI'][1], d['AMI'][2],
                                                                    d['ARI'][0], d['ARI'][1]))
        for k in ('ARI_entre_semillas_sdec_completo', 'ARI_permutado_vs_completo_misma_semilla',
                  'ARI_argmax_vs_ward_misma_semilla'):
            if k in o:
                print(k, 'mean %.3f (sd %.3f), n=%d' % o[k])


if __name__ == '__main__':
    main()
