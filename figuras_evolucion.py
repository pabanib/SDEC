# -*- coding: utf-8 -*-
"""
Evolution by region: Figure 6 (COVID-19), Figures 8 and 9 (wages and jobs, country and Cuyo) and the table of Cuyo.

A single diverging scale centered at zero and a common color bar are used in the three panels of each figure. The value
mapped is the regional mean of the standardized variable (z-score across departments for each variable and period, as it
enters the algorithm). The partitions are those mapped by figuras_mapas.py (figuras/particion_covid.csv and
figuras/particion_salarios.csv), so run that script first.
Usage:  python figuras_evolucion.py [covid|salarios]
Output: figuras/fig6.png, fig8.png, fig9.png, the regional means plotted (figuras/fig*_medias_regionales.csv) and
        figuras/tabla_cuyo.csv (annual means, 2014 to 2023, of the standardized wage bill per inhabitant of the
        departments of Cuyo in each region, and the number of departments n).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import constantes
from sdec import datos
from sdec.cartografia import flecha_y_escala, numeros_de_region

MESES_COVID = [4, 10, 16]              # month index (0 = March 2020): July 2020, January 2021, July 2021
ANIOS_SALARIOS = [2015, 2019, 2023]
PROV_CUYO = ['Mendoza', 'San Juan', 'San Luis']
CMAP = 'RdBu_r'
MESES = ['March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
         'January', 'February', 'March', 'April', 'May', 'June', 'July']
ANIO_MES = [2020] * 10 + [2021] * 7


def dibujar_evolucion(geo, grupos, valores, titulos, salida, etiqueta_barra, km, vmax=None, figsize=(15, 9.5),
                      fuente_num=8):
    """valores: (n, 3), the variable of each department in each panel. Returns (scale used, regional means)."""
    grupos = np.asarray(grupos)
    med = pd.DataFrame(valores).groupby(grupos).mean()
    med.columns = titulos
    if vmax is None:
        vmax = float(np.ceil(np.abs(med.values).max() * 2) / 2)          # rounded to 0.5
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(CMAP)
    gdf = gpd.GeoDataFrame({'grupo': grupos}, geometry=geo.values, index=geo.index)
    reg = gdf.dissolve(by='grupo')
    xmin, ymin, xmax, ymax = gdf.total_bounds
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for k, (ax, tit) in enumerate(zip(axes, titulos)):
        gdf['v'] = med[tit].reindex(gdf['grupo']).values
        gdf.plot(ax=ax, column='v', cmap=cmap, norm=norm, edgecolor='none')
        reg.boundary.plot(ax=ax, color='black', linewidth=0.8)
        numeros_de_region(ax, reg, fontsize=fuente_num, alpha=0.8)
        ax.set_title(tit, fontsize=13)
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if k == 0:
            flecha_y_escala(ax, xmin, xmax, ymin, ymax, km)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.035, pad=0.02, aspect=50)
    cb.set_label(etiqueta_barra, fontsize=11)
    fig.savefig(salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  saved', salida, '(symmetric scale +-%.1f)' % vmax)
    return vmax, med


def leer_particion(nombre, ids):
    p = pd.read_csv(os.path.join(constantes.DIR_FIGURAS, 'particion_%s.csv' % nombre)).set_index('link')
    return p.loc[ids, 'grupo'].astype(int).values


def covid():
    print('COVID-19 (Figure 6)')
    d = datos.datos_covid()
    g = leer_particion('covid', d['dep'].index)
    tit = ['%s %d' % (MESES[i], ANIO_MES[i]) for i in MESES_COVID]
    _, med = dibujar_evolucion(
        d['geo'], g, d['prop_st'][:, MESES_COVID], tit, os.path.join(constantes.DIR_FIGURAS, 'fig6.png'),
        'Regional mean of confirmed cases per inhabitant (z-score across departments, by month)', km=500)
    med.round(3).to_csv(os.path.join(constantes.DIR_FIGURAS, 'fig6_medias_regionales.csv'))


def salarios():
    print('Wages and jobs (Figures 8 and 9)')
    d = datos.datos_salarios()
    T = constantes.T_SAL
    g = leer_particion('salarios', d['dep'].index)
    anual = pd.DataFrame(d['prop_st'][:, :T], columns=d['periodos']).T.resample('YE').mean()
    anual.index = anual.index.year
    assert all(a in anual.index for a in ANIOS_SALARIOS), list(anual.index)
    vals = anual.loc[ANIOS_SALARIOS].values.T
    tit = [str(a) for a in ANIOS_SALARIOS]
    etiqueta = 'Regional mean of the wage bill per inhabitant (z-score across departments, annual mean)'
    vmax, med = dibujar_evolucion(d['geo'], g, vals, tit, os.path.join(constantes.DIR_FIGURAS, 'fig8.png'), etiqueta,
                                  km=500)
    med.round(3).to_csv(os.path.join(constantes.DIR_FIGURAS, 'fig8_medias_regionales.csv'))
    # Cuyo: same regions, means computed only with the departments of Cuyo, same scale as Figure 8
    filtro = d['dep']['provincia'].isin(PROV_CUYO).values
    _, med9 = dibujar_evolucion(d['geo'][filtro], g[filtro], vals[filtro], tit,
                                os.path.join(constantes.DIR_FIGURAS, 'fig9.png'),
                                etiqueta + '; Cuyo departments only', km=200, vmax=vmax, figsize=(15, 7.5), fuente_num=9)
    med9.round(3).to_csv(os.path.join(constantes.DIR_FIGURAS, 'fig9_medias_regionales_cuyo.csv'))
    # table of Cuyo: annual means 2014-2023 by region
    t = anual.T[filtro].groupby(g[filtro]).mean()
    t.insert(0, 'n', pd.Series(g[filtro]).value_counts().sort_index().values)
    t.index.name = 'region'
    t.round(2).to_csv(os.path.join(constantes.DIR_FIGURAS, 'tabla_cuyo.csv'))
    print(t.round(2).to_string())


def main():
    casos = [a for a in sys.argv[1:] if a in ('covid', 'salarios')] or ['covid', 'salarios']
    if 'covid' in casos:
        covid()
    if 'salarios' in casos:
        salarios()


if __name__ == '__main__':
    main()
