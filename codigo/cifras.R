# Cifras y tablas que leen las dos versiones del artículo, topologia.qmd y topologia_en.qmd, y la hoja de resumen.
# Cada una fija antes IDIOMA ("es" o "en"): de él dependen el formato de los números y el texto que estas funciones devuelven.
library(dplyr)
library(tidyr)
library(readr)
library(knitr)

source("codigo/rotulos.R")
DECIMAL <- tr(",", ".")
leer <- function(archivo) read_csv(file.path("resultados", archivo), show_col_types = FALSE)
# El espacio antes de % es de no separación: el signo no queda solo al cambiar de renglón.
pc <- function(x, d = 0) paste0(formatC(100 * x, format = "f", digits = d, decimal.mark = DECIMAL), tr("\u00a0%", "%"))
nm <- function(x, d = 2) formatC(x, format = "f", digits = d, decimal.mark = DECIMAL)
ent <- function(x) formatC(as.integer(x), format = "d", big.mark = tr(".", ","), decimal.mark = DECIMAL)
entre <- function(x) tr(paste0("«", x, "»", collapse = ", "), paste0("“", x, "”", collapse = ", "))

corpus <- "corpus/datos"
cifras <- with(read_csv(file.path(corpus, "resumen_corpus.csv"), show_col_types = FALSE), setNames(valor, clave))
corpus_por_anio <- read_csv(file.path(corpus, "corpus_por_anio.csv"), show_col_types = FALSE)
ventana <- function(desde, hasta, columna = "tesis") with(corpus_por_anio, sum(get(columna)[anio >= desde & anio <= hasta]))

CON_IA <- "con IA de por medio"
SIN_IA <- "sin IA de por medio"
mapas <- leer("som_contrastes.csv")
SEMILLAS <- n_distinct(mapas$semilla)
estabilidad <- leer("som_estabilidad.csv")
destacadas <- leer("som_destacadas.csv")
temas <- leer("temas_por_ventana.csv")
clasificacion <- leer("clasificacion_periodo.csv")
contrastes <- leer("homologia_contrastes.csv") |> mutate(valor_q = p.adjust(valor_p, method = "BH"))
potencia <- leer("homologia_potencia.csv")
sin_aprendizaje <- leer("homologia_sensibilidad.csv")
compacidad <- leer("homologia_compacidad.csv")
lsa <- with(leer("lsa.csv"), setNames(valor, clave))
vocabulario <- leer("vocabulario_resumen.csv")
cambios <- leer("vocabulario_cambios.csv")
marcadores <- leer("marcadores_por_anio.csv")
raices <- leer("marcadores_raices.csv")

mapa <- function(repr, comp) filter(mapas, representacion == repr, comparacion == comp)
auc <- function(comp, repr = "todas las palabras", clas = "perceptron", columna = "auc") {
  clasificacion |> filter(comparacion == comp, representacion == repr, clasificador == clas) |> pull(all_of(columna))
}
auc_ic <- function(comp, repr = "todas las palabras", clas = "perceptron") {
  paste0(nm(auc(comp, repr, clas)), " (", nm(auc(comp, repr, clas, "auc_inf")), tr(" a ", " to "), nm(auc(comp, repr, clas, "auc_sup")), ")")
}
voc <- function(comp, control_elegido = "todas las tesis") filter(vocabulario, comparacion == comp, control == control_elegido)
tasa_media <- function(desde, hasta, serie_elegida = "todas las raíces", area_elegida = "todas") {
  with(filter(marcadores, serie == serie_elegida, area == area_elegida, anio >= desde, anio <= hasta), weighted.mean(tasa, n_tesis))
}
ml <- function(v) with(filter(temas, ventana == v), de_aprendizaje_automatico / tesis)
suben_mas <- cambios |> filter(comparacion == CON_IA) |> slice_max(z, n = 15)
bajan_mas <- cambios |> filter(comparacion == CON_IA) |> slice_min(z, n = 15)
raices_ia <- filter(raices, comparacion == CON_IA)
raices_previas <- filter(raices, comparacion == SIN_IA)
propias <- leer("logodds_resumenes.csv")
mas_propias <- function(v, n = 6) entre(head(arrange(filter(propias, ventana == v), desc(abs(z)))$palabra, n))
# «Ninguna cambia» en vez de «0 cambian».
cuantas_cambian <- function(n) tr(
  ifelse(n == 0, "ninguna cambia", ifelse(n == 1, "1 cambia", paste(n, "cambian"))),
  ifelse(n == 0, "none changes", ifelse(n == 1, "1 changes", paste(n, "change")))
)
LSA <- "tesis (LSA)"
E5 <- "tesis (incrustaciones)"
ESCALA <- "diferencia de distancia media"
# Nombre de cada estadístico dentro de una frase.
PROSA <- setNames(
  tr(c("la distancia media", "$L_0$", "$L_1$", "$W_0$", "$W_1$", "$W_0$ a escala", "$W_1$ a escala"),
     c("the mean distance", "$L_0$", "$L_1$", "$W_0$", "$W_1$", "scaled $W_0$", "scaled $W_1$")),
  names(ESTADISTICOS)
)
p_de <- function(nube_elegida, comp, estadistico_elegido, columna = "valor_p", tabla = contrastes) {
  filter(tabla, nube == nube_elegida, comparacion == comp, estadistico == estadistico_elegido)[[columna]]
}
# Cambio relativo de la distancia media entre dos tesis de una ventana a la siguiente.
contraccion <- function(nube_elegida, comp = CON_IA, tabla = contrastes) {
  fila <- filter(tabla, nube == nube_elegida, comparacion == comp, estadistico == ESCALA)
  fila$observado / fila$nivel_de_la_primera
}
SIN_APRENDIZAJE <- unique(sin_aprendizaje$comparacion)
sobreviven <- filter(contrastes, valor_q < 0.05)
que_sobreviven <- function(nube_elegida) {
  combine_words(PROSA[filter(sobreviven, nube == nube_elegida)$estadistico], and = tr(" y ", " and "), oxford_comma = FALSE)
}

# Lo que el texto afirma de la forma de la nube. Si los datos cambian, esto falla y hay que reescribir esos párrafos.
stopifnot(
  length(SIN_APRENDIZAJE) == 1, nrow(potencia) > 0,
  # Lo que sobrevive al ajuste es con IA de por medio y en las nubes de tesis.
  all(sobreviven$comparacion == CON_IA), all(sobreviven$nube != "palabras"),
  # La contracción y la diferencia de ciclos sobreviven con las dos representaciones.
  all(c(p_de(LSA, CON_IA, ESCALA, "valor_q"), p_de(E5, CON_IA, ESCALA, "valor_q")) < 0.05),
  all(c(p_de(LSA, CON_IA, "W1", "valor_q"), p_de(E5, CON_IA, "W1", "valor_q")) < 0.05),
  contraccion(LSA) < 0, contraccion(E5) < 0,
  # Descontada la escala, la diferencia de ciclos no sobrevive con LSA y el control con incrustaciones da un p parecido.
  p_de(LSA, CON_IA, "W1 a escala", "valor_q") >= 0.05, p_de(E5, SIN_IA, "W1 a escala") < 0.05,
  # L1 baja con las incrustaciones.
  p_de(E5, CON_IA, "diferencia de L1", "observado") < 0, p_de(E5, CON_IA, "diferencia de L1", "valor_q") < 0.05,
  # Las tesis de aprendizaje automático están más juntas que un conjunto al azar, y sin ellas la contracción sigue.
  all(compacidad$valor_p < 0.05), all(compacidad$distancia_media < compacidad$distancia_media_al_azar),
  all(filter(sin_aprendizaje, estadistico == ESCALA)$valor_p < 0.05),
  abs(contraccion(LSA, SIN_APRENDIZAJE, sin_aprendizaje)) < abs(contraccion(LSA)),
  p_de(LSA, SIN_APRENDIZAJE, "W1", tabla = sin_aprendizaje) < 0.05, p_de(E5, SIN_APRENDIZAJE, "W1", tabla = sin_aprendizaje) >= 0.05,
  # Sin cambio plantado la prueba casi no rechaza.
  max(filter(potencia, fraccion_plantada == 0)$rechazos) <= 3,
  # Quitar las palabras de estilo no baja el AUC del clasificador.
  auc(CON_IA, "sin palabras de estilo") > auc(CON_IA) - 0.02,
  # El control del mapa no es nulo en todos los mapas, y el del vocabulario sí.
  sum(mapa("LSA", SIN_IA)$valor_p < 0.05) > 0, all(with(filter(vocabulario, comparacion == SIN_IA), suben + bajan) == 0)
)

# Tablas del artículo: las dos versiones las piden aquí para que un cambio llegue a las dos.
tabla_mapas <- function() {
  mapas |>
    group_by(representacion, comparacion) |>
    summarise(rechazan = paste0(sum(valor_p < 0.05), tr(" de ", " of "), n()), mediano = nm(median(valor_p), 3),
              extremos = paste0(nm(min(valor_p), 4), tr(" a ", " to "), nm(max(valor_p), 3)), .groups = "drop") |>
    arrange(representacion != "LSA", comparacion != SIN_IA) |>
    mutate(representacion = if_else(representacion == "LSA", tr("TF-IDF y LSA", "TF-IDF and LSA"), tr("Incrustaciones", "Embeddings")),
           comparacion = if_else(comparacion == SIN_IA, tr("Control (2014-2017 frente a 2018-2021)", "Control (2014-2017 vs. 2018-2021)"),
                                 tr("2018-2021 frente a 2023-2026", "2018-2021 vs. 2023-2026"))) |>
    setNames(tr(c("Representación", "Comparación", "Mapas con p < 0,05", "p mediano", "p mínimo y máximo"),
                c("Representation", "Comparison", "Maps with p < 0.05", "Median p", "Minimum and maximum p")))
}
tabla_unidades <- function() {
  destacadas |>
    transmute(letra, sentido = if_else(residuo > 0, tr("Crece", "Up"), tr("Baja", "Down")), terminos,
              `2014-2017`, `2018-2021`, `2023-2026`, residuo = nm(residuo, 1)) |>
    setNames(c(tr("Unidad", "Unit"), tr("Sentido", "Direction"), tr("Términos con más peso", "Terms with the most weight"),
               "2014-2017", "2018-2021", "2023-2026", tr("Residuo", "Residual")))
}
tabla_forma <- function() {
  con_ia <- tr("con IA", "with AI")
  contrastes |>
    filter(nube != "palabras") |>
    transmute(estadistico = factor(ESTADISTICOS[estadistico], levels = ESTADISTICOS),
              columna = paste0(if_else(nube == LSA, "LSA", "E5"), ", ", if_else(comparacion == CON_IA, con_ia, "control")),
              celda = paste0(nm(valor_p, 4), " (", nm(valor_q, 3), ")")) |>
    pivot_wider(names_from = columna, values_from = celda) |>
    arrange(estadistico) |>
    select(estadistico, all_of(paste0(c("LSA", "LSA", "E5", "E5"), ", ", c(con_ia, "control")))) |>
    rename(!!tr("Estadístico", "Statistic") := estadistico)
}
tabla_plantado <- function() {
  potencia |>
    transmute(estadistico = factor(ESTADISTICOS[estadistico], levels = ESTADISTICOS),
              fraccion = if_else(fraccion_plantada == 0, tr("Sin cambio", "No change"), paste0(pc(fraccion_plantada), tr(" plantado", " planted"))),
              celda = paste0(rechazos, tr(" de ", " of "), replicas)) |>
    pivot_wider(names_from = fraccion, values_from = celda) |>
    arrange(estadistico) |>
    rename(!!tr("Estadístico", "Statistic") := estadistico)
}
tabla_vocabulario <- function() {
  en_ingles <- c("todas las tesis" = "All theses", "longitud fija" = "Fixed length", "Estadística" = "Statistics",
                 "Matemáticas" = "Mathematics", "sin aprendizaje automático" = "Without machine learning")
  vocabulario |>
    transmute(incluidas = tr(paste0(toupper(substr(control, 1, 1)), substring(control, 2)), unname(en_ingles[control])), comparacion,
              celda = paste0(tr("suben ", "up "), suben, tr(", bajan ", ", down "), bajan, " (", n_tesis_a, tr(" y ", " and "), n_tesis_b, ")")) |>
    pivot_wider(names_from = comparacion, values_from = celda) |>
    transmute(incluidas, control = .data[[SIN_IA]], con_ia = .data[[CON_IA]]) |>
    setNames(tr(c("Tesis incluidas", "Control: 2016-2018 frente a 2019-2021", "2019-2021 frente a 2023-2026"),
                c("Theses included", "Control: 2016-2018 vs. 2019-2021", "2019-2021 vs. 2023-2026")))
}
