# Rótulos visibles del artículo y de las figuras en la lengua que fija IDIOMA ("es" o "en").
# Los resultados guardan sus rótulos en español; en inglés se traducen al mostrarlos.

stopifnot(IDIOMA %in% c("es", "en"))
tr <- function(es, en) if (IDIOMA == "es") es else en

# Nombre de cada estadístico de forma en las tablas y en la figura de homología.
ESTADISTICOS <- setNames(
  tr(c("Distancia media (diferencia)", "L0, componentes (diferencia)", "L1, ciclos (diferencia)", "W0, componentes", "W1, ciclos", "W0 a escala", "W1 a escala"),
     c("Mean distance (difference)", "L0, components (difference)", "L1, cycles (difference)", "W0, components", "W1, cycles", "Scaled W0", "Scaled W1")),
  c("diferencia de distancia media", "diferencia de L0", "diferencia de L1", "W0", "W1", "W0 a escala", "W1 a escala")
)
