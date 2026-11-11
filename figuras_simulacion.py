# -*- coding: utf-8 -*-
"""
Figures 2, 3 and 4: target regions and regionalizations of the two simulated scenarios.

The labels of each method are aligned with the target regions (Hungarian assignment on the contingency table), so the
same color means the same target region in all panels; each panel shows its AMI with the target.
Partitions: SDEC, the run with the median AMI among the seeds of resultados/simulaciones_sdec.jsonl; SH, computed here;
SKATER, REDCAP and SCHC from resultados/comparadores_escenario<k>.csv (comparadores.R); if they do not exist, those panels
are left empty.
Usage:  python figuras_simulacion.py     Output: figuras/fig2.png, fig3.png, fig4.png and fig3_ami.csv, fig4_ami.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import geopandas as gpd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score as ami
import constantes
from sdec import datos
from sdec.cartografia import PALETA_OBJETIVO as PALETA, flecha_y_escala
from sdec.experimentos import leer_jsonl
from dgp_simulacion import escenario

NOMBRES = ['Northwest (without Tucumán)', 'Northeast', 'Cuyo', 'Pampeana', 'Patagonia', 'Tucumán']


def alinear(obj, lab):
    """Relabels `lab` with the numbers 1..6 of the target regions."""
    lab = np.asarray(lab)
    u = list(np.unique(lab))
    C = np.zeros((6, len(u)))
    for i in range(6):
        for j, v in enumerate(u):
            C[i, j] = np.sum((obj == i + 1) & (lab == v))
    fi, cj = linear_sum_assignment(-C)
    mapa = {u[j]: i + 1 for i, j in zip(fi, cj)}
    for k, v in enumerate([v for v in u if v not in mapa]):
        mapa[v] = 7 + k
    return np.array([mapa[v] for v in lab])


def panel(ax, geo, etiquetas, titulo, con_escala=False):
    gdf = gpd.GeoDataFrame({'g': etiquetas}, geometry=geo.values)
    gdf['color'] = [PALETA[g - 1] if g <= 6 else '#999999' for g in gdf['g']]
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='white', linewidth=0.15)
    gdf.dissolve(by='g').boundary.plot(ax=ax, color='black', linewidth=0.6)
    ax.set_title(titulo, fontsize=12)
    ax.set_axis_off()
    ax.set_aspect('equal')
    if con_escala:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        flecha_y_escala(ax, xmin, xmax, ymin, ymax)


def leyenda(fig):
    h = [mp.Patch(facecolor=PALETA[i], edgecolor='black', label='%d. %s' % (i + 1, NOMBRES[i])) for i in range(6)]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=10, frameon=False)


def sdec_mediana(obj, esc):
    filas = [f for f in leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'simulaciones_sdec.jsonl'))
             if f['escenario'] == esc]
    if not filas:
        return None
    a = np.array([ami(obj, f['grupos']) for f in filas])
    i = int(np.argsort(a)[len(a) // 2])
    print('  SDEC scenario %d: %d runs, median AMI %.4f (range %.4f to %.4f); seed %s is plotted' %
          (esc, len(a), np.median(a), a.min(), a.max(), filas[i]['semilla']))
    return np.asarray(filas[i]['grupos'])


def main():
    dep = datos.departamentos()
    geo = dep.geometry
    obj = datos.objetivo_simulacion(dep)
    W = datos.matriz_vecinos(dep, constantes.KNN_SIM)
    fig, ax = plt.subplots(figsize=(6, 9.5))
    panel(ax, geo, obj, 'Target regions', con_escala=True)
    leyenda(fig)
    fig.savefig(os.path.join(constantes.DIR_FIGURAS, 'fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    for esc, n_fig in ((1, 3), (2, 4)):
        X = escenario(obj, esc, constantes.SEMILLA_DATOS)
        met = [('SDEC', sdec_mediana(obj, esc)),
               ('SH', AgglomerativeClustering(constantes.K_SIM, connectivity=W).fit(X).labels_)]
        ruta = os.path.join(constantes.DIR_RESULTADOS, 'comparadores_escenario%d.csv' % esc)
        r = pd.read_csv(ruta) if os.path.exists(ruta) else None
        for m in ('skater', 'redcap', 'schc'):
            met.append((m.upper(), r[m].values if r is not None else None))
        fig, axes = plt.subplots(2, 3, figsize=(11, 13))
        axes = axes.ravel()
        panel(axes[0], geo, obj, 'Target', con_escala=True)
        filas = []
        for ax, (nm, lab) in zip(axes[1:], met):
            if lab is None:
                ax.set_axis_off()
                ax.set_title('%s (not available)' % nm)
                continue
            a = ami(obj, lab)
            panel(ax, geo, alinear(obj, lab), '%s (AMI = %.2f)' % (nm, a))
            filas.append({'escenario': esc, 'metodo': nm, 'ami': a})
        leyenda(fig)
        fig.subplots_adjust(hspace=0.12, wspace=0.05, bottom=0.08)
        fig.savefig(os.path.join(constantes.DIR_FIGURAS, 'fig%d.png' % n_fig), dpi=300, bbox_inches='tight')
        plt.close(fig)
        pd.DataFrame(filas).to_csv(os.path.join(constantes.DIR_FIGURAS, 'fig%d_ami.csv' % n_fig), index=False)
        print(pd.DataFrame(filas).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
