# -*- coding: utf-8 -*-
"""
Maps of the regions: Figure 5 (COVID-19) and Figure 7 (wages and jobs).

Each region is identified by its number written on the map, and the colors are assigned with a graph coloring so that
two regions that touch never share a color. Black lines delimit regions and dotted lines delimit provinces.
Partitions:
  COVID-19 : by default the partition provided in datos/covid_particion_figura5.csv (the one shown in Figures 5 and 6).
             With the argument `corridas`, the most representative of the runs of resultados/covid_semillas.jsonl
             (highest mean ARI against the others).
  Wages    : the most representative of the runs of resultados/salarios_semillas.jsonl.
Usage:  python figuras_mapas.py [covid|salarios] [corridas]
Output: figuras/fig5.png, figuras/fig7.png and the plotted partition of each case (figuras/particion_*.csv).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import constantes
from sdec import datos
from sdec.cartografia import (PALETA_REGIONES, flecha_y_escala, adyacencia, colorear, numeros_de_region,
                              mas_representativa)
from sdec.experimentos import leer_jsonl


def dibujar(geo, provincia, grupos, salida):
    gdf = gpd.GeoDataFrame({'grupo': grupos}, geometry=geo.values, index=geo.index)
    reg = gdf.dissolve(by='grupo')
    ids = list(reg.index)
    col = colorear(adyacencia([reg.geometry[i] for i in ids], tol=2000.0))
    ncol = max(col.values()) + 1
    print('  %d regions, %d colors needed so that no touching regions repeat a color' % (len(ids), ncol))
    assert ncol <= len(PALETA_REGIONES), 'more colors are needed in PALETA_REGIONES'
    color_de = {ids[i]: PALETA_REGIONES[col[i]] for i in range(len(ids))}
    fig, ax = plt.subplots(figsize=(8, 15))
    gdf['color'] = [color_de[g] for g in gdf['grupo']]
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='white', linewidth=0.25)
    prov = gpd.GeoDataFrame({'p': provincia.values}, geometry=geo.values).dissolve(by='p')
    prov.boundary.plot(ax=ax, color='#555555', linewidth=0.5, linestyle=':')
    reg.boundary.plot(ax=ax, color='black', linewidth=1.1)
    numeros_de_region(ax, reg)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    flecha_y_escala(ax, xmin, xmax, ymin, ymax, grande=True)
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.savefig(salida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  saved', salida)


def particion_covid(d, corridas):
    if corridas:
        return mas_representativa(leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'covid_semillas.jsonl')))[1]
    p = pd.read_csv(os.path.join(constantes.DIR_DATOS, 'covid_particion_figura5.csv')).set_index('link')
    return p.loc[d['dep'].index, 'grupo'].astype(int).values


def main():
    args = sys.argv[1:]
    casos = [a for a in args if a in ('covid', 'salarios')] or ['covid', 'salarios']
    corridas = 'corridas' in args
    if 'covid' in casos:
        print('COVID-19')
        d = datos.datos_covid()
        g = particion_covid(d, corridas)
        assert len(g) == len(d['dep'])
        dibujar(d['geo'], d['dep']['provincia'], g, os.path.join(constantes.DIR_FIGURAS, 'fig5.png'))
        pd.DataFrame({'link': d['dep'].index, 'grupo': g}).to_csv(
            os.path.join(constantes.DIR_FIGURAS, 'particion_covid.csv'), index=False)
    if 'salarios' in casos:
        print('Wages and jobs')
        d = datos.datos_salarios()
        filas = leer_jsonl(os.path.join(constantes.DIR_RESULTADOS, 'salarios_semillas.jsonl'))
        sem, g = mas_representativa(filas)
        assert len(g) == len(d['dep'])
        dibujar(d['geo'], d['dep']['provincia'], g, os.path.join(constantes.DIR_FIGURAS, 'fig7.png'))
        pd.DataFrame({'link': d['dep'].index, 'grupo': g, 'semilla': sem}).to_csv(
            os.path.join(constantes.DIR_FIGURAS, 'particion_salarios.csv'), index=False)


if __name__ == '__main__':
    main()
