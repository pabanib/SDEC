# -*- coding: utf-8 -*-
"""
Figure 1: diagram of SDEC with the actual dimensions of the network (sdec/deep_cluster.py, gen_modelo): two dense
branches T -> 80 -> 40 -> 10 (one per variable), concatenation (20), 50, linear latent layer of 6, decoder 40 -> 80 -> 2T;
clustering layer (Student's t, K centroids) on the latent layer; target P (neighbor-smoothed, updated every 20 iterations);
KL(P||Q) loss; final partition: Ward constrained by W applied to Q.
Usage:  python figura1.py        Output: figuras/fig1.png (300 dpi)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import constantes
SALIDA = constantes.DIR_FIGURAS

AZUL, VERDE, NARANJA, GRIS = '#dbe7f5', '#dff0e0', '#fbe5cc', '#eeeeee'
BORDE = '#444444'


def caja(ax, x, y, w, h, texto, color, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=color, ec=BORDE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', linespacing=1.25)


def flecha(ax, x0, y0, x1, y1, texto=None, estilo='-|>', ls='-', dy=0.18):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=11, lw=1.0,
                                 color=BORDE, linestyle=ls, shrinkA=0, shrinkB=0))
    if texto:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, texto, ha='center', va='bottom', fontsize=7.5, style='italic')


def main():
    os.makedirs(SALIDA, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 25)
    ax.set_ylim(-1.6, 14)
    ax.axis('off')

    # ---- encoder: dos ramas densas
    y1, y2 = 10.0, 4.8                      # base de cada rama (alto de caja 1.4)
    for y, nombre in ((y1, 'Variable 1'), (y2, 'Variable 2')):
        caja(ax, 0.2, y - 0.1, 2.2, 1.6, nombre + '\n$T$ periods', GRIS, fs=8.5)
        xs = [3.1, 5.3, 7.5]
        for x, n in zip(xs, [80, 40, 10]):
            caja(ax, x, y, 1.6, 1.4, 'Dense\n%d\nReLU' % n, AZUL, fs=8)
        flecha(ax, 2.4, y + 0.7, 3.1, y + 0.7)
        flecha(ax, 4.7, y + 0.7, 5.3, y + 0.7)
        flecha(ax, 6.9, y + 0.7, 7.5, y + 0.7)
    cy = (y1 + y2) / 2 + 0.7
    caja(ax, 10.2, cy - 1.0, 1.7, 2.0, 'Concat.\n(20)', AZUL, fs=8.5)
    flecha(ax, 9.1, y1 + 0.7, 10.2, cy + 0.6)
    flecha(ax, 9.1, y2 + 0.7, 10.2, cy - 0.6)
    caja(ax, 12.4, cy - 0.7, 1.5, 1.4, 'Dense\n50\nReLU', AZUL, fs=8)
    flecha(ax, 11.9, cy, 12.4, cy)
    caja(ax, 14.4, cy - 0.85, 2.2, 1.7, 'Latent layer\n$h=6$\n(linear)', VERDE, fs=8.5, bold=True)
    flecha(ax, 13.9, cy, 14.4, cy)
    ax.text(5.5, 12.1, 'Encoder: one dense branch per variable, then joint layers', ha='center', fontsize=9, style='italic')

    # ---- decodificador (solo en el preentrenamiento)
    yd = 1.4
    caja(ax, 14.1, yd, 1.6, 1.4, 'Dense\n40\nReLU', GRIS, fs=8)
    caja(ax, 11.8, yd, 1.6, 1.4, 'Dense\n80\nReLU', GRIS, fs=8)
    caja(ax, 9.0, yd, 2.0, 1.4, 'Output $2T$\n(linear)', GRIS, fs=8)
    flecha(ax, 15.5, cy - 0.85, 14.9, yd + 1.4)
    flecha(ax, 14.1, yd + 0.7, 13.4, yd + 0.7)
    flecha(ax, 11.8, yd + 0.7, 11.0, yd + 0.7)
    ax.text(11.0, 0.6, 'Decoder (pre-training only): reconstructs the $2T$ inputs;\ncosine-similarity loss, 100 epochs',
            ha='center', fontsize=8.3, style='italic')

    # ---- capa de clustering, objetivo y perdida
    caja(ax, 17.7, cy - 0.85, 3.0, 1.7, 'Clustering layer\n$K$ centroids\n(Student-$t$ kernel)', NARANJA, fs=8.5, bold=True)
    flecha(ax, 16.6, cy, 17.7, cy)
    caja(ax, 22.0, cy - 0.85, 2.3, 1.7, 'KL$(P\\,\\|\\,Q)$\nloss', NARANJA, fs=9, bold=True)
    flecha(ax, 20.7, cy, 22.0, cy)
    ty = 11.3
    caja(ax, 17.7, ty, 6.6, 1.9, 'Target $P$ (updated every 20 iterations):\n'
                                 '$p_{ij}\\propto\\,\\left(\\mathbf{W}\\,[\\,q_{ij}^{2}/\\sum_{i} q_{ij}\\,]\\right)_{ij}$', NARANJA, fs=8.5)
    flecha(ax, 19.2, cy + 0.85, 19.2, ty, 'soft assignments $Q$', dy=-0.1)
    ax.texts[-1].set_ha('left'); ax.texts[-1].set_x(19.4)
    flecha(ax, 23.15, ty, 23.15, cy + 0.85, '$P$', dy=-0.1)
    ax.texts[-1].set_ha('left'); ax.texts[-1].set_x(23.35)
    ax.text(21.3, cy - 1.6, 'KL gradient updates the encoder\nand the centroids (up to 1,000 iterations)',
            ha='center', va='top', fontsize=8, style='italic')

    # ---- particion final
    caja(ax, 17.7, 0.7, 6.6, 2.6, 'Final partition\nWard agglomerative clustering applied to $Q$,\n'
                                  'constrained by the $k$-nearest-neighbor\n'
                                  'graph $\\mathbf{W}$ (recomputed at each update)', GRIS, fs=8.5, bold=False)
    ax.texts[-1].set_fontweight('normal')
    flecha(ax, 18.6, cy - 0.85, 18.6, 3.3)
    ax.text(18.8, 4.0, '$Q$', fontsize=8.5, style='italic', va='center')

    ax.text(0.2, -1.5, '$T$ = number of periods (100 in the simulations, 17 in COVID-19, 108 in salaries); $K$ = number of regions (6, 18 and 20); '
                      '$h=6$ in all cases.\n$\\mathbf{W}$: $k$-nearest-neighbor spatial weights ($k=6$; $k=4$ in salaries). Source: own elaboration.',
            fontsize=7.8, va='bottom', ha='left')
    fig.savefig(os.path.join(SALIDA, 'fig1.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print('written', os.path.join(SALIDA, 'fig1.png'))


if __name__ == '__main__':
    main()
