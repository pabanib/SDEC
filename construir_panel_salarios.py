# -*- coding: utf-8 -*-
"""
Builds the department x month panel of the wages and jobs application from the source files and compares it with the panel
provided in datos/salarios_panel.csv.

Source files (not distributed; see README): 
  w_mean_depto_total_clae2.csv       mean wage by department, month and activity (clae2)     [columns fecha, codigo_departamento_indec, clae2, w_mean]
  puestos_depto_total_por_clae2.csv  registered jobs by department, month and activity       [columns fecha, codigo_departamento_indec, clae2, puestos]
  cer-uva-uvi-diarios.csv            daily CER index                                           [columns indice_tiempo, cer_diario]
  clae_agg_corr.csv                  correspondence clae2 -> letter of the activity (separator ';', columns clae2, letra)
Treatment:
  wage bill = jobs * mean wage (the value -99 of the sources, statistical secrecy, is taken as 0);
  wage bill deflated by the CER (last daily value of each month);
  SAC adjustment: the deflated wage bill of June and December is divided by 1.5 (jobs are not adjusted);
  jobs and deflated wage bill of the department = sum over activities (jobs by letter A to S and Z, as in the sources);
  period: January 2014 to November 2023 (119 months); departments: those of datos/salarios_panel.csv.
Usage:  python construir_panel_salarios.py <dir_csv> <cer.csv> <clae_agg_corr.csv>
Output: resultados/salarios_panel_reconstruido.csv (same columns as datos/salarios_panel.csv) and the maximum absolute
        difference against the provided panel.
"""
import os
import sys
import numpy as np
import pandas as pd
import constantes

LETRAS = list('ABCDEFGHIJKLMNOPQRS') + ['Z']
FIN = pd.Timestamp('2023-12-01')


def main():
    dir_csv, arch_cer, arch_clae = sys.argv[1:4]
    ref = pd.read_csv(os.path.join(constantes.DIR_DATOS, 'salarios_panel.csv'))
    links = sorted(ref['link'].unique())
    s = pd.read_csv(os.path.join(dir_csv, 'w_mean_depto_total_clae2.csv'), parse_dates=['fecha'])
    p = pd.read_csv(os.path.join(dir_csv, 'puestos_depto_total_por_clae2.csv'), parse_dates=['fecha'])
    claves = ['fecha', 'codigo_departamento_indec', 'clae2']
    b = s.merge(p[claves + ['puestos']], how='left', on=claves, validate='one_to_one').replace(-99, 0)
    b['puestos'] = b['puestos'].fillna(0)
    b['masa'] = b['puestos'] * b['w_mean']
    b = b[b.fecha < FIN]
    cer = pd.read_csv(arch_cer, parse_dates=['indice_tiempo']).set_index('indice_tiempo')['cer_diario']
    cer_m = cer.resample('ME').last().loc['2014-01-01':FIN.strftime('%Y-%m-%d')]
    cer_m.index = cer_m.index.to_period('M').to_timestamp()
    meses = cer_m.index[cer_m.index < FIN]
    assert len(meses) == 119, len(meses)
    clae = pd.read_csv(arch_clae, sep=';', encoding='utf-8-sig')[['clae2', 'letra']].drop_duplicates()
    assert clae.clae2.is_unique, 'a clae2 with more than one letter'
    b = b.merge(clae, on='clae2', how='left')
    b['letra'] = b['letra'].str.strip().fillna('Z')
    b['masa_cer'] = b['masa'] / b.fecha.map(cer_m) / np.where(b.fecha.dt.month.isin([6, 12]), 1.5, 1.0)
    dep = b[b.codigo_departamento_indec.isin(links)]
    idx = pd.MultiIndex.from_product([links, meses], names=['link', 'fecha'])
    let = dep.pivot_table(index=['codigo_departamento_indec', 'fecha'], columns='letra', values='puestos',
                          aggfunc='sum', fill_value=0).reindex(idx, fill_value=0).reindex(columns=LETRAS, fill_value=0)
    tot = dep.groupby(['codigo_departamento_indec', 'fecha'])['masa_cer'].sum().reindex(idx, fill_value=0)
    out = pd.DataFrame({'puestos': let.sum(axis=1), 'masa_cer': tot}, index=idx).reset_index()
    out['mes'] = out.fecha.dt.strftime('%Y-%m')
    out = out[['link', 'mes', 'puestos', 'masa_cer']]
    out.to_csv(os.path.join(constantes.DIR_RESULTADOS, 'salarios_panel_reconstruido.csv'), index=False)
    m = out.merge(ref, on=['link', 'mes'], suffixes=('', '_ref'))
    print('panel:', out.shape, '| departments', out.link.nunique(), '| months', out.mes.nunique(), '| rows matched', len(m))
    for v in ('puestos', 'masa_cer'):
        print('maximum absolute difference in %s against datos/salarios_panel.csv: %.3e' %
              (v, np.abs(m[v] - m[v + '_ref']).max()))


if __name__ == '__main__':
    main()
