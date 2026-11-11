# -*- coding: utf-8 -*-
"""
Runs the whole analysis in order (about 180 SDEC runs of 3 to 5 minutes each on a laptop CPU, that is, 10 to 15 hours,
plus the 200 runs of the Monte Carlo comparison and the 200 runs of its deep graph-based methods, about 10 more hours).
Usage:  python ejecutar_todo.py [--rapido]
With --rapido a reduced version is run (2 seeds per experiment and 2 draws of the Monte Carlo comparison, to check
that the installation works).
The steps that need R (comparadores.R) are run only if Rscript is available. Each step can also be run on its own.
"""
import shutil
import subprocess
import sys

RAPIDO = '--rapido' in sys.argv
S_SIM, S_ABL, N_APP = (['1', '2'], ['1', '2'], '2') if RAPIDO else (None, None, '30')
N_MC = '2' if RAPIDO else '100'


def paso(*args):
    print('\n>>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable] + list(args), check=True)


def main():
    paso('exportar_para_R.py')
    if shutil.which('Rscript'):
        print('\n>>> Rscript comparadores.R', flush=True)
        subprocess.run(['Rscript', 'comparadores.R'], check=True)
    else:
        print('\nRscript not found: comparadores.R is skipped (SKATER, REDCAP and SCHC will not be available).')
    for esc in ('1', '2'):
        paso('correr_simulaciones.py', esc, *(S_SIM or []))
        for k in ('4', '5', '6'):
            paso('correr_sensibilidad_k.py', esc, k, *(S_SIM or []))
        paso('correr_ablaciones.py', esc, *(S_ABL or []))
        paso('correr_ablacion_lineal.py', esc, *(S_ABL or []))
    paso('control_suavizado.py')
    paso('tablas_simulaciones.py')
    paso('analizar_ablaciones.py')
    for a in ('0.5', '1.0'):
        paso('correr_montecarlo.py', a, N_MC)
        paso('correr_montecarlo_profundos.py', a, N_MC)
    paso('tablas_montecarlo.py')
    paso('correr_covid.py', 'semillas', N_APP)
    paso('correr_covid.py', 'repetir', '42')
    paso('correr_salarios.py', 'semillas', N_APP)
    paso('correr_salarios.py', 'repetir', '42')
    for f in ('figura1.py', 'figuras_simulacion.py', 'figuras_apendice.py', 'figuras_mapas.py', 'figuras_evolucion.py'):
        paso(f)


if __name__ == '__main__':
    main()
