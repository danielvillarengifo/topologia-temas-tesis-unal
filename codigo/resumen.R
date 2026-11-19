# Resumen y palabras clave del artículo en la lengua de IDIOMA, con las cifras de cifras.R.
# Lo leen topologia.qmd, topologia_en.qmd y la hoja de solo resumen, topologia_resumen.qmd.

resumen <- function() tr(
  glue::glue(
    "Se estudia si las tesis de maestría y doctorado en Estadística y Matemáticas de la Universidad Nacional de Colombia ",
    "cambiaron de temas, de forma y de vocabulario desde 2023. El corpus son {ent(cifras['tesis_de_analisis'])} resúmenes y, ",
    "de {ent(cifras['tesis_con_texto_corrido'])} tesis, también el texto corrido (la prosa del PDF, sin tablas ni fórmulas). ",
    "Se comparan ventanas de años: 2018-2021 frente a 2023-2026, con la inteligencia artificial (IA) generativa de por medio, ",
    "y como control dos ventanas anteriores entre sí. En mapas autoorganizados entrenados con {SEMILLAS} semillas, ",
    "las tesis de 2023-2026 se reparten distinto que las de 2018-2021 en {sum(mapa('LSA', CON_IA)$valor_p < 0.05)} de {SEMILLAS} mapas ",
    "con una representación del texto y en {sum(mapa('incrustaciones', CON_IA)$valor_p < 0.05)} de {SEMILLAS} con la otra; ",
    "en el control, en {sum(mapa('LSA', SIN_IA)$valor_p < 0.05)} y en {sum(mapa('incrustaciones', SIN_IA)$valor_p < 0.05)}. ",
    "Un perceptrón multicapa distingue esas dos ventanas con un área bajo la curva ROC (AUC) de {nm(auc(CON_IA))}; ",
    "en el control el AUC es {nm(auc(SIN_IA))} (0,5 es el azar). Las tesis de aprendizaje automático pasan de ",
    "{pc(ml('2018-2021'))} a {pc(ml('2023-2026'))} del total. Vistas como una nube de puntos, las tesis recientes se parecen ",
    "más entre sí: la distancia media entre dos tesis bajó {pc(-contraccion(LSA), 1)} con una representación y ",
    "{pc(-contraccion(E5), 1)} con la otra, más de lo que da barajar las tesis entre ventanas ",
    "(p de {nm(p_de(LSA, CON_IA, ESCALA), 4)} y {nm(p_de(E5, CON_IA, ESCALA), 4)}). La homología persistente encuentra además ",
    "que los ciclos de esa nube (cadenas cerradas de tesis parecidas) difieren entre las dos ventanas; descontada esa ",
    "contracción, la evidencia es débil. En el control ningún contraste de forma sobrevive al ajuste por multiplicidad. ",
    "En el texto corrido, {voc(CON_IA)$suben} de {ent(voc(CON_IA)$palabras_probadas)} palabras suben de frecuencia entre ",
    "2019-2021 y 2023-2026 y en el control {cuantas_cambian(voc(SIN_IA)$suben + voc(SIN_IA)$bajan)}; con todas las tesis ",
    "reducidas a la misma longitud suben {voc(CON_IA, 'longitud fija')$suben}. Todas las comparaciones son temporales: ",
    "no aíslan el efecto de la IA."
  ),
  glue::glue(
    "We study whether the master's and doctoral theses in Statistics and Mathematics at the Universidad Nacional de Colombia ",
    "changed in topics, shape and vocabulary from 2023 onward. The corpus is {ent(cifras['tesis_de_analisis'])} abstracts and, ",
    "for {ent(cifras['tesis_con_texto_corrido'])} theses, also the running text (the prose of the PDF, without tables or formulas). ",
    "We compare windows of years: 2018-2021 against 2023-2026, with generative artificial intelligence (AI) in between, ",
    "and, as a control, two earlier windows against each other. In self-organizing maps trained with {SEMILLAS} seeds, ",
    "the theses of 2023-2026 are distributed differently from those of 2018-2021 in {sum(mapa('LSA', CON_IA)$valor_p < 0.05)} ",
    "of {SEMILLAS} maps with one text representation and in {sum(mapa('incrustaciones', CON_IA)$valor_p < 0.05)} of {SEMILLAS} ",
    "with the other; in the control, in {sum(mapa('LSA', SIN_IA)$valor_p < 0.05)} and in {sum(mapa('incrustaciones', SIN_IA)$valor_p < 0.05)}. ",
    "A multilayer perceptron separates those two windows with an area under the ROC curve (AUC) of {nm(auc(CON_IA))}; ",
    "in the control the AUC is {nm(auc(SIN_IA))} (0.5 is chance). Machine learning theses go from {pc(ml('2018-2021'))} ",
    "to {pc(ml('2023-2026'))} of the total. Seen as a point cloud, recent theses are more alike: the mean distance between ",
    "two theses fell by {pc(-contraccion(LSA), 1)} with one representation and {pc(-contraccion(E5), 1)} with the other, ",
    "a larger drop than random reassignment of theses to windows produces ",
    "(p = {nm(p_de(LSA, CON_IA, ESCALA), 4)} and {nm(p_de(E5, CON_IA, ESCALA), 4)}). ",
    "Persistent homology also finds that the cycles of that cloud (closed chains of similar theses) differ between the two windows; ",
    "once this contraction is discounted, that evidence is weak. In the control no shape test survives the multiplicity adjustment. ",
    "In the running text, {voc(CON_IA)$suben} of {ent(voc(CON_IA)$palabras_probadas)} words rise in frequency between 2019-2021 ",
    "and 2023-2026, while in the control {cuantas_cambian(voc(SIN_IA)$suben + voc(SIN_IA)$bajan)}; with all theses cut to the ",
    "same length, {voc(CON_IA, 'longitud fija')$suben} rise. All comparisons are temporal: they do not isolate the effect of AI."
  )
)

PALABRAS_CLAVE <- tr(
  "mapa autoorganizado, perceptrón multicapa, homología persistente, análisis topológico de datos, transformadores, minería de texto, IA generativa",
  "self-organizing map, multilayer perceptron, persistent homology, topological data analysis, transformers, text mining, generative AI"
)
