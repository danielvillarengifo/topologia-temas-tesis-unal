# Dibuja las figuras del artículo de topología a partir de resultados/.
#
# Lee: som_unidades.csv, som_destacadas.csv, clasificacion_periodo.csv, homologia_contrastes.csv,
#   marcadores_por_anio.csv, vocabulario_cambios.csv y logodds_resumenes.csv.
# Escribe figuras/fig_*.png.
#
# Uso: Rscript figuras.R <carpeta resultados> <carpeta figuras>

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(ggwordcloud)
})

args <- commandArgs(trailingOnly = TRUE)
carpeta <- args[1]
destino <- args[2]
dir.create(destino, showWarnings = FALSE)

AZUL <- "#0072B2"
NARANJA <- "#D55E00"
GRIS <- "#7F7F7F"
COLOR_COMPARACION <- c("sin IA de por medio" = GRIS, "con IA de por medio" = NARANJA)
PALABRAS_POR_LADO <- 15
SEMILLA <- 20261118

decimal <- function(x, cifras = 2) formatC(x, format = "f", digits = cifras, decimal.mark = ",")

theme_set(
  theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 11),
      plot.title.position = "plot",
      plot.subtitle = element_text(colour = "grey30"),
      plot.caption = element_text(colour = "grey40", hjust = 0),
      plot.caption.position = "plot",
      panel.grid.minor = element_blank(),
      legend.position = "top",
      panel.spacing = unit(1.1, "lines"),
      strip.text = element_text(face = "bold", hjust = 0)
    )
)

# ggplot no parte los títulos largos: se parten aquí al ancho de la figura.
envolver <- function(texto, ancho) paste(strwrap(texto, ancho), collapse = "\n")
rotulos <- function(title = NULL, subtitle = NULL, caption = NULL, ..., ancho = 72) {
  labs(
    title = if (!is.null(title)) envolver(title, ancho),
    subtitle = if (!is.null(subtitle)) envolver(subtitle, ancho * 1.35),
    caption = if (!is.null(caption)) envolver(caption, ancho * 1.55),
    ...
  )
}

leer <- function(archivo) read_csv(file.path(carpeta, archivo), show_col_types = FALSE)
guardar <- function(grafico, nombre, ancho = 6.5, alto = 4) {
  ggsave(file.path(destino, nombre), grafico, width = ancho, height = alto, dpi = 300, bg = "white")
}

# Mapa de temas
unidades <- leer("som_unidades.csv")
destacadas <- leer("som_destacadas.csv")
fig_mapa <- ggplot(unidades, aes(columna, fila)) +
  geom_tile(aes(fill = residuo), colour = "white", linewidth = 0.8) +
  geom_text(data = destacadas, aes(label = letra), fontface = "bold", size = 3.6) +
  scale_fill_gradient2(
    low = AZUL, mid = "grey95", high = NARANJA, midpoint = 0, na.value = "white",
    breaks = c(min(unidades$residuo, na.rm = TRUE), 0, max(unidades$residuo, na.rm = TRUE)),
    labels = c("Menos", "Lo esperado", "Más"),
    name = "Tesis de 2023-2026 frente a 2018-2021"
  ) +
  coord_equal() +
  scale_y_reverse() +
  rotulos(
    title = "Uno de los veinte mapas de temas, con las unidades donde más cambian las tesis recientes",
    subtitle = "Cada cuadro es una unidad del mapa autoorganizado; las unidades vecinas tienen temas parecidos",
    ancho = 56
  ) +
  theme(axis.title = element_blank(), axis.text = element_blank(), panel.grid = element_blank(),
        legend.key.width = unit(1.6, "cm"), legend.title.position = "top")
guardar(fig_mapa, "fig_mapa.png", alto = 5.8)

# Clasificación de la ventana
clasificacion <- leer("clasificacion_periodo.csv") |>
  filter(comparacion != "diferencia") |>
  mutate(
    clasificador = if_else(clasificador == "perceptron", "Perceptrón multicapa", "Regresión logística"),
    comparacion = factor(comparacion, levels = names(COLOR_COMPARACION)),
    representacion = factor(representacion, levels = c("todas las palabras", "sin palabras de estilo", "incrustaciones", "composición"),
                            labels = c("TF-IDF y LSA", "TF-IDF y LSA sin palabras de estilo", "Incrustaciones", "Solo área, nivel, sede y longitud"))
  )
fig_clasificacion <- ggplot(clasificacion, aes(auc, clasificador, colour = comparacion)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  geom_linerange(aes(xmin = auc_inf, xmax = auc_sup), position = position_dodge(width = 0.5)) +
  geom_point(size = 2, position = position_dodge(width = 0.5)) +
  geom_text(aes(label = decimal(auc), x = auc_sup), hjust = -0.3, size = 2.7, position = position_dodge(width = 0.5), show.legend = FALSE) +
  facet_wrap(~representacion, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = COLOR_COMPARACION, name = NULL, labels = function(x) paste("Ventanas", x)) +
  scale_x_continuous(limits = c(0.4, 0.9), labels = function(x) decimal(x, 1)) +
  rotulos(
    title = "Qué tan bien se predice la ventana de una tesis por su resumen",
    subtitle = "Área bajo la curva ROC (AUC) al predecir la ventana de una tesis, fuera de pliegue, con intervalo del 95 %",
    x = "AUC (0,5 es el azar)", y = NULL
  )
guardar(fig_clasificacion, "fig_clasificacion.png", alto = 4.6)

# Homología persistente: dónde cae cada estadístico observado frente a su distribución nula
ESTADISTICOS <- c(
  "diferencia de distancia media" = "Distancia media (diferencia)", "diferencia de L0" = "L0, componentes (diferencia)",
  "diferencia de L1" = "L1, ciclos (diferencia)", "W0" = "W0, componentes", "W1" = "W1, ciclos",
  "W0 a escala" = "W0 a escala", "W1 a escala" = "W1 a escala"
)
contrastes <- leer("homologia_contrastes.csv") |>
  mutate(
    comparacion = factor(comparacion, levels = names(COLOR_COMPARACION)),
    estadistico = factor(ESTADISTICOS[estadistico], levels = rev(ESTADISTICOS)),
    nube = factor(nube, levels = c("tesis (LSA)", "tesis (incrustaciones)", "palabras"),
                  labels = c("Tesis, LSA", "Tesis, incrustaciones", "Palabras")),
    # Los estadísticos tienen unidades distintas: se dibuja la posición del observado
    # en su distribución nula, con 0 en el centro y -1 y 1 en los percentiles 2,5 y 97,5.
    posicion = (observado - (nula_inf + nula_sup) / 2) / ((nula_sup - nula_inf) / 2)
  )
stopifnot(!anyNA(contrastes$estadistico), !anyNA(contrastes$nube))
fig_homologia <- ggplot(contrastes, aes(posicion, estadistico, colour = comparacion)) +
  annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, fill = "grey90") +
  geom_vline(xintercept = 0, colour = "white") +
  geom_point(size = 2, position = position_dodge(width = 0.55)) +
  facet_wrap(~nube, nrow = 1) +
  scale_colour_manual(values = COLOR_COMPARACION, name = NULL, labels = function(x) paste("Ventanas", x)) +
  scale_x_continuous(labels = scales::label_number(decimal.mark = ",")) +
  rotulos(
    title = "Dónde cae cada estadístico observado frente a barajar las tesis entre ventanas",
    subtitle = "Franja gris: del percentil 2,5 al 97,5 de la distribución nula, llevada a la escala de -1 a 1",
    x = "Posición del valor observado en su distribución nula", y = NULL
  )
guardar(fig_homologia, "fig_homologia.png", ancho = 7, alto = 3.9)

# Raíces marcadoras por año
raices <- leer("marcadores_raices.csv") |> distinct(raiz, con_uso_tecnico)
SERIES <- c("todas las raíces" = sprintf("Las %d raíces", nrow(raices)), "sin uso técnico" = "Sin las raíces de uso técnico")
marcadores <- leer("marcadores_por_anio.csv") |>
  filter(area %in% c("todas", "Estadística", "Matemáticas")) |>
  mutate(
    area = factor(area, levels = c("todas", "Estadística", "Matemáticas"), labels = c("Todas las tesis", "Estadística", "Matemáticas")),
    serie = factor(SERIES[serie], levels = SERIES)
  )
fig_marcadores <- ggplot(marcadores, aes(anio, tasa, colour = serie)) +
  geom_linerange(aes(ymin = inf, ymax = sup), alpha = 0.6, position = position_dodge(width = 0.4)) +
  geom_line(position = position_dodge(width = 0.4), linewidth = 0.4) +
  geom_point(size = 1.3, position = position_dodge(width = 0.4)) +
  facet_wrap(~area, ncol = 1) +
  scale_colour_manual(values = setNames(c(GRIS, NARANJA), SERIES), name = NULL) +
  scale_x_continuous(breaks = 2016:2026) +
  scale_y_continuous(limits = c(0, NA)) +
  rotulos(
    title = "Tasa de raíces marcadoras en el texto corrido, por año y área",
    subtitle = "Apariciones por cada 10.000 palabras de texto corrido, media por tesis con intervalo del 95 %",
    x = "Año de publicación", y = "Apariciones por 10.000 palabras",
    caption = sprintf("Raíces marcadoras: %d raíces que las guías de estilo asocian al texto de modelos de lenguaje; %d tienen además uso técnico en estadística y matemáticas.", nrow(raices), sum(raices$con_uso_tecnico))
  )
guardar(fig_marcadores, "fig_marcadores.png", alto = 4.8)

# Palabras que más suben y bajan en el texto corrido
cambios <- leer("vocabulario_cambios.csv") |> filter(comparacion == "con IA de por medio")
extremas <- bind_rows(slice_max(cambios, z, n = PALABRAS_POR_LADO), slice_min(cambios, z, n = PALABRAS_POR_LADO)) |>
  mutate(tipo = if_else(es_marcadora, "Raíz marcadora", "Otra palabra"))
fig_palabras <- ggplot(extremas, aes(log2_razon, reorder(palabra, log2_razon))) +
  geom_vline(xintercept = 0, colour = "grey50") +
  geom_segment(aes(x = 0, xend = log2_razon, yend = palabra), linewidth = 0.4, colour = "grey60") +
  geom_point(aes(colour = tipo), size = 2) +
  scale_colour_manual(values = c("Raíz marcadora" = NARANJA, "Otra palabra" = "grey25"), name = NULL) +
  scale_x_continuous(labels = function(x) paste0("×", decimal(2^x, 1))) +
  rotulos(
    title = "Las palabras que más suben y más bajan en el texto corrido",
    subtitle = sprintf("Razón suavizada entre la tasa de 2023-2026 y la de 2019-2021; las %d con mayor estadístico en cada sentido", PALABRAS_POR_LADO),
    x = "Cuántas veces más frecuente en 2023-2026", y = NULL
  )
guardar(fig_palabras, "fig_palabras.png", alto = 4.8)

# Nubes de palabras de los resúmenes
propias <- leer("logodds_resumenes.csv") |>
  mutate(ventana = factor(ventana, levels = c("2018-2021", "2023-2026")))
set.seed(SEMILLA)
fig_nube <- ggplot(propias, aes(label = palabra, size = abs(z), colour = ventana)) +
  geom_text_wordcloud(family = "sans") +
  facet_wrap(~ventana, labeller = as_labeller(c("2018-2021" = "Más propias de 2018-2021", "2023-2026" = "Más propias de 2023-2026"))) +
  scale_size_area(max_size = 6) +
  scale_colour_manual(values = c("2018-2021" = GRIS, "2023-2026" = NARANJA), guide = "none") +
  rotulos(
    title = "Las 40 palabras más propias de los resúmenes de cada ventana",
    subtitle = "El tamaño es la razón de momios ponderada; los tamaños se comparan dentro de cada panel"
  )
guardar(fig_nube, "fig_nube.png", alto = 3.8)
