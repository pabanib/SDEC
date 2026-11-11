# -*- coding: utf-8 -*-
"""Paths and constants shared by all scripts."""
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(RAIZ, 'datos')
DIR_RESULTADOS = os.path.join(RAIZ, 'resultados')
DIR_FIGURAS = os.path.join(RAIZ, 'figuras')
EPSG = 5345                    # POSGAR 2007 / Argentina 3 (coordinates in meters)
SEMILLA_DATOS = 3264           # seed of the simulated data
K_SIM, KNN_SIM = 6, 6          # regions and neighbors in the simulations
K_COVID, KNN_COVID, T_COVID = 18, 6, 17
K_SAL, KNN_SAL, T_SAL = 20, 4, 108

os.makedirs(DIR_RESULTADOS, exist_ok=True)
os.makedirs(DIR_FIGURAS, exist_ok=True)
