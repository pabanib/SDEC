# SDEC: replication code

Code and data to replicate the simulations, the two applications (COVID-19 and wages and jobs in Argentina), the
ablations, the tables and the figures of the paper "Redefining Regions in Space and Time: A Deep Learning Method for
Spatio-Temporal Clustering" (Spatial Deep Embedded Clustering, SDEC).

## Contents

```
sdec/                     the method and its tools
  deep_cluster.py           SDEC: autoencoder, clustering layer, spatial target, constrained Ward step
  profundos.py              deep graph-based comparison methods: SDCN (Bo et al., 2020) and graph autoencoder (Kipf and Welling, 2016)
  region2vec.py             Region2Vec (Liang et al., 2022, 2025), adapted to data without flows
  series.py                 DTW and shape-based distances, Ward-like clustering on dissimilarities with or without the
                            contiguity constraint (ClustGeo), and the spatiotemporal clustering of Morelli et al. (2025)
  metricas.py               RBTSS, cosine similarity, silhouette, Calinski-Harabasz, Davies-Bouldin
  datos.py                  loading of the data and construction of the standardized matrices and neighbor matrices
  experimentos.py           seeding, training and refinement helpers shared by the experiment scripts
  aplicaciones.py           runs, summaries and contiguity checks of the two applications
  cartografia.py            common elements of the maps
constantes.py             paths, number of regions, neighbors and periods of each case
dgp_simulacion.py         data generating process of the simulations
dgp_alternativo.py        alternative data generating process (serial and spatial dependence in the noise, random
                          connected target partitions), not used to develop SDEC
datos/                    input data (see below)
resultados/               outputs of the scripts (created when they run)
figuras/                  figures and tables produced by the scripts
```

## Environment

Python 3.13 with the packages of `requirements.txt` (TensorFlow 2.21 / Keras 3.15, scikit-learn, geopandas, libpysal).
No GPU is needed. The comparison methods SKATER, REDCAP and SCHC (Tables 3 and 4) are run with the R package `rgeoda`
(`comparadores.R`; needs R with the packages `sf` and `rgeoda`). In the Monte Carlo comparison (Table 2) they are run
with the Python package `pygeoda`, built on the same library; without it, `correr_montecarlo.py` skips those three
methods and says so.

All scripts are run from the folder that contains this file, for example `python correr_simulaciones.py 1`.

## Data (`datos/`)

* `departamentos.gpkg`: polygons of the 525 departments of Argentina (EPSG:4326), with the code `link` (province and
  department code), the name of the department and of the province, and the population (`personas`).
* `covid_panel.csv`: confirmed cases and deaths of each department and month, March 2020 to July 2021 (17 months).
* `salarios_panel.csv`: registered jobs (`puestos`) and wage bill deflated by the CER and adjusted for the SAC
  (`masa_cer`) of each department and month, January 2014 to November 2023 (119 months), for the 504 departments with
  data. The wage bill is the product of the jobs and the mean wage; the values coded as -99 in the sources (statistical secrecy) are taken as
  zero; the CER is the last daily value of each month; the deflated wage bill of June and December is divided by 1.5.
* `covid_particion_figura5.csv`: partition of the 525 departments into 18 regions shown in Figures 5 and 6. It is an
  input of `figuras_mapas.py` and `figuras_evolucion.py`: since the partition of a run depends on the environment (see
  below), it is provided so that those two figures can be redrawn exactly. With the argument `corridas`,
  `figuras_mapas.py` draws instead the most representative run of those saved by `correr_covid.py`.

The raw source files of the wages and jobs panel are not distributed. `construir_panel_salarios.py` rebuilds the panel
from them and compares it with `salarios_panel.csv`.

## Order of the scripts

| Step | Script | Produces |
|---|---|---|
| 1 | `python correr_simulaciones.py <1 or 2>` | SDEC in each simulated scenario, seeds 1 to 10 (Tables 3 and 4) |
| 2 | `python exportar_para_R.py`, `Rscript comparadores.R` | SKATER, REDCAP and SCHC (Tables 3 and 4) |
| 3 | `python tablas_simulaciones.py` | extrinsic metrics (AMI, MI, homogeneity, Rand), SH, summary of the sensitivity to k |
| 4 | `python correr_sensibilidad_k.py <1 or 2> <4, 5 or 6>` | sensitivity to the number of neighbors |
| 5 | `python correr_ablaciones.py <1 or 2>` | variants (i) to (ix) of Table 5, five seeds |
| 6 | `python correr_ablacion_lineal.py <1 or 2>` | variants (xi) and (xii) of Table 5, five seeds |
| 7 | `python control_suavizado.py` | variant (x) of Table 5 |
| 8 | `python analizar_ablaciones.py` | Table 5 and the temporal permutation test |
| 8b | `python correr_montecarlo.py <0.5 or 1.0> 100`, `python correr_montecarlo_profundos.py <0.5 or 1.0> 100`, `python tablas_montecarlo.py` | Monte Carlo comparison over 100 draws of the data per scenario, all methods on each draw; the second script adds SDCN and the graph autoencoder on the same draws (Table 2) |
| 8c | `python correr_montecarlo_series.py <0.5 or 1.0> 100` | DTW and SBD with constrained Ward, k-Shape and Morelli et al. (2025) on the draws of Table 2 |
| 8d | `python correr_montecarlo_region2vec.py <0.5 or 1.0> 100` | Region2Vec on the draws of Table 2 |
| 8e | `python ajustar_comparadores.py ajuste <0.5 or 1.0>`, then `python ajustar_comparadores.py evaluar <0.5 or 1.0>` | tuning of SDCN, the graph autoencoder and Region2Vec on 10 independent draws, and evaluation of the selected configurations on the draws of Table 2 |
| 8f | `python correr_ablacion_montecarlo.py <0.5 or 1.0> 30` | key variants of the ablation, including neighbors + own term, on 30 draws |
| 8g | `python correr_montecarlo_alternativo.py <0.5 or 1.0> aleatorio 100` | all methods on the alternative data generating process |
| 8h | `python tablas_montecarlo_nuevos.py` | paired comparison of SDEC with the methods of steps 8c to 8e |
| 8i | `python rbtss_comparadores.py` | RBTSS of the comparison methods in the applications and of the true partition in the simulations |
| 9 | `python correr_covid.py semillas 30` | COVID-19, K = 18, k = 6, 30 seeds (Table of metrics, contiguity, sizes) |
| 10 | `python correr_salarios.py semillas 30` | wages and jobs, K = 20, k = 4, 30 seeds |
| 11 | `python correr_covid.py repetir 42`, `python correr_salarios.py repetir 42` | same seed run twice in the same process |
| 12 | `python figura1.py`, `figuras_simulacion.py`, `figuras_apendice.py`, `figuras_mapas.py`, `figuras_evolucion.py` | Figures 1 to 9, A1 and A2, and `figuras/tabla_cuyo.csv` |

`python ejecutar_todo.py` runs everything in order (`--rapido` runs two seeds of each experiment to check the
installation). The complete set is about 180 SDEC runs of 3 to 5 minutes each on a laptop CPU, plus the 200 runs of the Monte Carlo
comparison (about 6 hours) and the 200 runs of its deep graph-based methods (about 4 hours); in both, the two scenarios
write to different files and can run at the same time. Every script that runs
several seeds saves each one as soon as it ends and, if interrupted, resumes where it stopped.

## Reproducibility

Seeds fix the initialization of the networks, the k-means and the order of the mini-batches, and two runs with the same
seed in the same software environment give the same partition. Across environments (hardware, operating system and
versions of TensorFlow and of the numerical libraries) the floating-point operations of the training are not performed in
the same order, and since the refinement runs up to 1,000 iterations the partition of a given seed can change, mainly
in the applications. The partitions of each seed, and therefore the values of the tables, are consequently not
bit-for-bit reproducible across machines; what is stable are the orders of magnitude and the comparisons between
methods and variants made within the same environment. For that reason `correr_ablacion_lineal.py` trains the
nonlinear and the linear networks in the same run, and the 30 seeds of the applications are used to report the
variability. The Monte Carlo comparison (Table 2) was run on Windows with Python 3.12; the methods without random
initialization give the same values there and in the environment above, whereas SDEC, SDCN and the graph autoencoder
depend on the environment for a given seed.

Things that this package does not reproduce by code: the figure of the partition shown in Figures 5 and 6 is drawn from
the partition provided in `datos/`; the raw files of the wages panel are not included; and the R step needs `rgeoda`.
