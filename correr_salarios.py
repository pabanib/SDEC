# -*- coding: utf-8 -*-
"""
Wages and jobs application (Section 5): SDEC with K = 20 regions, 108 months, 4-nearest-neighbor matrix.

Usage:
    python correr_salarios.py semillas [n]   runs the seeds 1..n (default 30), saving each one as soon as it ends
                                         (resultados/salarios_semillas.jsonl; if interrupted, it resumes where it
                                         stopped), and then writes resultados/salarios_resumen.json
    python correr_salarios.py resumen        only the summary across the saved runs
    python correr_salarios.py repetir [s]    runs the seed s (default 42) twice and compares the partitions
Each run takes about 5 minutes on a CPU.
"""
import sys
from sdec import aplicaciones

if __name__ == '__main__':
    aplicaciones.main('salarios', sys.argv[1:])
