# -*- coding: utf-8 -*-
"""
Data generating process of the simulations (Section 4).

For each department i with target region r(i) and each period t = 0, ..., 99:
    y_it^v = max{0, f_r(i)^v(t) + e_i^v},  e_i^v ~ U(-a/2, a/2) drawn once per unit and variable,
with a = 0.5 (scenario 1) and a = 1 (scenario 2), seed 3264. Each column of the resulting n x 200 matrix is
standardized across departments. The curves are those of Table 1 of the paper.
"""
import numpy as np

FUNC1 = {1: lambda x: 0.14 - ((x - 50) / 100) ** 2, 2: lambda x: 0.25 - ((x - 30) / 100) ** 2,
         3: lambda x: 0.20 - ((x - 80) / 100) ** 2, 4: lambda x: 0.10 - ((x - 50) / 100) ** 2,
         5: lambda x: 0.3 - ((x - 50) / 100) ** 2, 6: lambda x: 0.3 - ((x - 50) / 100) ** 2}
FUNC2 = {1: lambda x: 0.14 - ((2 * x - 90) / 160) ** 2, 2: lambda x: 0.25 - ((2 * x - 100) / 160) ** 2,
         3: lambda x: 0.20 - ((2 * x - 150) / 160) ** 2, 4: lambda x: 0.20 - ((2 * x - 100) / 160) ** 2,
         5: lambda x: 0.15 - ((2 * x - 120) / 160) ** 2, 6: lambda x: 0.3 - ((2 * x - 120) / 160) ** 2}

AMPLITUD = {1: 0.5, 2: 1.0}     # amplitude a of each scenario


def genera_datos(obj, amplitud, semilla, periodos=100):
    """obj: target region (1..6) of each unit. Returns the standardized matrix (n, 2 * periodos)."""
    np.random.seed(semilla)
    p = np.arange(0, periodos)
    v1 = np.array([FUNC1[r](p) + (np.random.rand() - 0.5) * amplitud for r in obj])
    v2 = np.array([FUNC2[r](p) + (np.random.rand() - 0.5) * amplitud for r in obj])
    v = np.c_[v1 * (v1 > 0), v2 * (v2 > 0)]
    return (v - v.mean(0)) / v.std(0)


def escenario(obj, esc, semilla=3264):
    """Data of scenario 1 or 2."""
    return genera_datos(obj, AMPLITUD[esc], semilla)
