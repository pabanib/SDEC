# -*- coding: utf-8 -*-
"""Common elements of the maps: palettes, north arrow, scale bar, coloring of regions and region numbers."""
import numpy as np

PALETA_REGIONES = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999', '#8DD3C7',
                   '#BEBADA']
PALETA_OBJETIVO = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#0072B2', '#D55E00']   # target regions 1..6


def flecha_y_escala(ax, xmin, xmax, ymin, ymax, km=500, grande=False):
    ancho, alto = xmax - xmin, ymax - ymin
    f = 1.4 if grande else 1.0
    x0 = xmax - (0.08 if grande else 0.07) * ancho
    y1 = ymax - 0.015 * alto
    y0 = y1 - 0.085 * alto
    ax.annotate('', xy=(x0, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5 * f, mutation_scale=16 * f))
    ax.text(x0, y1 + 0.01 * alto, 'N', ha='center', va='bottom', fontsize=11 * f, fontweight='bold')
    L = km * 1000.0
    xs1 = xmax - (0.04 if grande else 0.02) * ancho
    xs0 = xs1 - L
    ys = ymin + (0.03 if grande else 0.02) * alto
    ax.plot([xs0, xs1], [ys, ys], color='black', lw=3 * f, solid_capstyle='butt')
    ax.text((xs0 + xs1) / 2, ys + 0.01 * alto, '%d km' % km, ha='center', va='bottom', fontsize=9 * f)


def adyacencia(regiones, tol):
    """Two regions are adjacent if their geometries, widened by `tol` meters, intersect."""
    gb = [g.buffer(tol) for g in regiones]
    n = len(gb)
    vec = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if gb[i].intersects(gb[j]):
                vec[i].add(j)
                vec[j].add(i)
    return vec


def colorear(vec):
    """DSATUR: each region takes the smallest color not used by its neighbors."""
    n = len(vec)
    color = {}
    while len(color) < n:
        sin = [i for i in range(n) if i not in color]
        sat = {i: len({color[j] for j in vec[i] if j in color}) for i in sin}
        i = max(sin, key=lambda k: (sat[k], len(vec[k])))
        usados = {color[j] for j in vec[i] if j in color}
        c = 0
        while c in usados:
            c += 1
        color[i] = c
    return color


def numeros_de_region(ax, reg, fontsize=10, alpha=0.85):
    """Writes the number of each region on its largest part (fragments under 10% of it are not labeled)."""
    for i, g in zip(reg.index, reg.geometry):
        partes = list(g.geoms) if hasattr(g, 'geoms') else [g]
        amax = max(p.area for p in partes)
        for parte in partes:
            if parte.area < 0.10 * amax:
                continue
            p = parte.representative_point()
            ax.text(p.x, p.y, str(i), ha='center', va='center', fontsize=fontsize, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=alpha, edgecolor='none'))


def mas_representativa(filas):
    """Run with the highest mean ARI against the other runs. Returns (seed, labels)."""
    from sklearn.metrics import adjusted_rand_score as ari
    g = [np.asarray(f['grupos']) for f in filas]
    M = np.array([[ari(a, b) for b in g] for a in g])
    np.fill_diagonal(M, np.nan)
    medias = np.nanmean(M, 1)
    i = int(np.argmax(medias))
    print('  most representative run: seed %d (mean ARI against the others = %.3f; range %.3f to %.3f)' %
          (filas[i]['semilla'], medias[i], medias.min(), medias.max()))
    return filas[i]['semilla'], np.asarray(filas[i]['grupos'])
