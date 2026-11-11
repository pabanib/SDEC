# -*- coding: utf-8 -*-
"""Loading of the data, construction of the standardized matrices and of the neighbor matrices."""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
from sklearn.preprocessing import StandardScaler
import constantes


def departamentos():
    """The 525 departments, indexed by link, sorted by link, in EPSG:5345.
    Columns: codpcia, departamen, provincia, personas (population), geometry."""
    g = gpd.read_file(os.path.join(constantes.DIR_DATOS, 'departamentos.gpkg'))
    g = g.set_index('link').sort_index().to_crs(epsg=constantes.EPSG)
    return g


def matriz_vecinos(geo, k):
    """Sparse k-nearest-neighbor matrix (KNN on the centroids of the polygons)."""
    df = gpd.GeoDataFrame(index=geo.index, geometry=geo.geometry.values)
    return KNN.from_dataframe(df, k=k).sparse


def objetivo_simulacion(dep):
    """Target regions of the simulation (1..6): the five traditional regions of Argentina,
    with Tucuman as a sixth region."""
    regiones = {'noroeste': ('Jujuy', 'Salta', 'Tucumán', 'Santiago del Estero', 'Catamarca'),
                'noreste': ('Formosa', 'Chaco', 'Misiones', 'Corrientes', 'Santa Fe', 'Entre Ríos'),
                'cuyo': ('La Rioja', 'San Juan', 'Mendoza', 'San Luis'),
                'pampeana': ('Córdoba', 'La Pampa', 'Buenos Aires', 'Ciudad Autónoma de Buenos Aires'),
                'patagonia': ('Neuquén', 'Río Negro', 'Chubut', 'Santa Cruz', 'Tierra del Fuego')}
    reg = []
    for p in dep['provincia']:
        hit = [j for j, (k, l) in enumerate(regiones.items(), start=1) if p in l]
        assert len(hit) == 1, p
        reg.append(hit[0])
    reg = np.array(reg)
    reg[(dep['provincia'] == 'Tucumán').values] = 6
    return reg


def datos_covid():
    """COVID-19: confirmed cases and deaths per inhabitant, 17 months (March 2020 to July 2021).
    Returns a dict with prop (n x 2T), prop_st (standardized by column), W (KNN, k = 6), geo, dep."""
    dep = departamentos()
    p = pd.read_csv(os.path.join(constantes.DIR_DATOS, 'covid_panel.csv'))
    conf = p.pivot(index='link', columns='mes', values='confirmados').reindex(dep.index)
    fall = p.pivot(index='link', columns='mes', values='fallecidos').reindex(dep.index)
    assert conf.shape[1] == constantes.T_COVID
    X = np.c_[conf.values, fall.values]
    prop = X / dep['personas'].values.reshape(-1, 1)
    prop_st = StandardScaler().fit_transform(prop)
    W = matriz_vecinos(dep, constantes.KNN_COVID)
    return {'prop': prop, 'prop_st': prop_st, 'W': W, 'geo': dep.geometry, 'dep': dep,
            'meses': list(conf.columns)}


def datos_salarios(k=constantes.KNN_SAL):
    """Wages and registered jobs: wage bill (deflated by the CER, SAC-adjusted) and number of jobs, both divided by
    the population of the department, 12-month moving average, standardized by column (variable-period).
    Returns a dict with prop (n x 2T), prop_st, W (KNN), geo, dep, periodos (T months)."""
    dep = departamentos()
    s = pd.read_csv(os.path.join(constantes.DIR_DATOS, 'salarios_panel.csv'))
    s = s.set_index(['link', 'mes']).sort_index(level=0)
    r = s.groupby(level=0)[['masa_cer', 'puestos']].rolling(12).mean()
    r.index = r.index.droplevel(0)
    r = r.dropna()
    w = r[['masa_cer', 'puestos']].unstack()          # columns: (variable, month); masa_cer first
    ids = w.index
    dep = dep.loc[ids]
    assert w.shape[1] == 2 * constantes.T_SAL, w.shape
    prop = (w.values + 1) / (dep['personas'].values.reshape(-1, 1) + 1)
    prop_st = (prop - prop.mean(0)) / prop.std(0)
    W = matriz_vecinos(dep, k)
    periodos = pd.to_datetime(w['masa_cer'].columns)
    return {'prop': prop, 'prop_st': prop_st, 'W': W, 'geo': dep.geometry, 'dep': dep, 'periodos': periodos}
