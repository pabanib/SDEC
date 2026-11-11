# SKATER, REDCAP and SCHC (rgeoda) on the simulated scenarios, with K = 6 regions and a 6-nearest-neighbor weights
# matrix, using the 200 simulated columns of each department as variables.
# Run after exportar_para_R.py, from the folder of this file:   Rscript comparadores.R
# Output: resultados/comparadores_escenario1.csv and resultados/comparadores_escenario2.csv
# (columns: skater, redcap, schc; one row per department, in the order of the GeoPackages).
library(sf)
library(rgeoda)

for (esc in 1:2) {
  g <- st_read(file.path("resultados", sprintf("escenario%d.gpkg", esc)), quiet = TRUE)
  w <- knn_weights(g, 6)
  vars <- st_drop_geometry(g)
  sk <- skater(k = 6, w = w, df = vars)
  rc <- redcap(k = 6, w = w, df = vars)
  sc <- schc(k = 6, w = w, df = vars)
  out <- data.frame(skater = sk$Clusters, redcap = rc$Clusters, schc = sc$Clusters)
  write.csv(out, file.path("resultados", sprintf("comparadores_escenario%d.csv", esc)), row.names = FALSE)
}
