# -*- coding: utf-8 -*-
"""
Exports the data of the two simulated scenarios as GeoPackages (525 departments, columns '0' to '199' and the polygon
geometry, EPSG:5345), the input of comparadores.R (SKATER, REDCAP and SCHC of the R package rgeoda).
Usage:  python exportar_para_R.py        Writes resultados/escenario1.gpkg and resultados/escenario2.gpkg
"""
import os
import geopandas as gpd
import constantes
from sdec import datos
from dgp_simulacion import escenario


def main():
    dep = datos.departamentos()
    obj = datos.objetivo_simulacion(dep)
    for esc in (1, 2):
        X = escenario(obj, esc, constantes.SEMILLA_DATOS)
        g = gpd.GeoDataFrame(X, columns=[str(i) for i in range(X.shape[1])], geometry=dep.geometry.values,
                             crs=dep.crs)
        ruta = os.path.join(constantes.DIR_RESULTADOS, 'escenario%d.gpkg' % esc)
        g.to_file(ruta, driver='GPKG')
        print('written', ruta, X.shape)


if __name__ == '__main__':
    main()
