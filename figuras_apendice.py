# -*- coding: utf-8 -*-
"""
Appendix figures: provinces and Autonomous City of Buenos Aires (Figure A1) and traditional macro-regions of Argentina
(Figure A2), drawn from the polygons of the 525 departments (datos/departamentos.gpkg).
Usage:  python figuras_apendice.py        Output: figuras/figA1.png, figuras/figA2.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import constantes
from sdec import datos
from sdec.cartografia import flecha_y_escala

MACRO = {'Noroeste': ('Jujuy', 'Salta', 'Tucumán', 'Santiago del Estero', 'Catamarca'),
         'Noreste': ('Formosa', 'Chaco', 'Misiones', 'Corrientes', 'Santa Fe', 'Entre Ríos'),
         'Cuyo': ('La Rioja', 'San Juan', 'Mendoza', 'San Luis'),
         'Pampeana': ('Córdoba', 'La Pampa', 'Buenos Aires', 'Ciudad Autónoma de Buenos Aires'),
         'Patagonia': ('Neuquén', 'Río Negro', 'Chubut', 'Santa Cruz', 'Tierra del Fuego')}
# label position: (dx, dy) as a fraction of the extent when the label is displaced from the representative point
DESPLAZADAS = {'Ciudad Autónoma de Buenos Aires': (0.12, 0.0, 'CABA'), 'Catamarca': (-0.30, 0.0, 'Catamarca')}


def mapa(geo, grupos, etiquetas, salida, desplazadas=None):
    gdf = gpd.GeoDataFrame({'g': grupos}, geometry=geo.values).dissolve(by='g')
    fig, ax = plt.subplots(figsize=(6, 12))
    gdf.plot(ax=ax, color='#d3d3d3', edgecolor='black', linewidth=0.8)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    for nombre, g in zip(gdf.index, gdf.geometry):
        partes = list(g.geoms) if hasattr(g, 'geoms') else [g]
        p = max(partes, key=lambda q: q.area).representative_point()
        txt = etiquetas.get(nombre, nombre)
        if desplazadas and nombre in desplazadas:
            dx, dy, txt = desplazadas[nombre]
            ax.annotate(txt, xy=(p.x, p.y), xytext=(p.x + dx * (xmax - xmin), p.y + dy * (ymax - ymin)),
                        fontsize=10, fontweight='bold', va='center', arrowprops=dict(arrowstyle='->', color='black'))
        else:
            ax.text(p.x, p.y, txt, ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))
    flecha_y_escala(ax, xmin, xmax, ymin, ymax, km=500, grande=True)
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.savefig(salida, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', salida)


def main():
    dep = datos.departamentos()
    mapa(dep.geometry, dep['provincia'].values, {'Ciudad Autónoma de Buenos Aires': 'CABA'},
         os.path.join(constantes.DIR_FIGURAS, 'figA1.png'), desplazadas=DESPLAZADAS)
    macro = np.array([next(k for k, v in MACRO.items() if p in v) for p in dep['provincia']])
    mapa(dep.geometry, macro, {}, os.path.join(constantes.DIR_FIGURAS, 'figA2.png'))


if __name__ == '__main__':
    main()
