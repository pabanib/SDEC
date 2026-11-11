# -*- coding: utf-8 -*-
"""
COVID-19 application (Section 5): SDEC with K = 18 regions, 17 months, 6-nearest-neighbor matrix.

Usage:
    python correr_covid.py semillas [n]   runs the seeds 1..n (default 30), saving each one as soon as it ends
                                         (resultados/covid_semillas.jsonl; if interrupted, it resumes where it
                                         stopped), and then writes resultados/covid_resumen.json
    python correr_covid.py resumen        only the summary across the saved runs
    python correr_covid.py repetir [s]    runs the seed s (default 42) twice and compares the partitions
Each run takes about 3 minutes on a CPU.
"""
import sys
from sdec import aplicaciones

if __name__ == '__main__':
    aplicaciones.main('covid', sys.argv[1:])
