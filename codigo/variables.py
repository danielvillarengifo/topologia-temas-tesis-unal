"""Variables estilométricas de un texto académico en español.

Todas son tasas o promedios, de modo que no dependen de la longitud del texto,
y se escogieron por depender poco del tema: palabras de función, conectores,
terminaciones, puntuación y forma de las oraciones. Las terminaciones se
cuentan por su forma: «-ando» incluye «cuando» y «-imos» incluye «mínimos».
"""
import re
import unicodedata

import numpy as np
import pandas as pd

PALABRAS_FUNCION = (
    "de la que el en y a los del se las por un para con no una su al lo como más o pero sus le ya este esta "
    "entre cuando muy sin sobre también hasta donde desde todo nos durante uno ni otros ese eso ante e esto "
    "antes algunos unos otro otras otra tanto esa estos mucho cual poco estas algunas algo son es fue ser ha "
    "han sido está están puede pueden cada dos así bajo dicho dicha tal tales cuyo cuya mediante según tras "
    "hacia aquí"
).split()

CONECTORES = [
    "además", "asimismo", "sin embargo", "no obstante", "por lo tanto", "por tanto", "por otro lado",
    "por otra parte", "finalmente", "en este sentido", "en este contexto", "en particular", "a través de",
    "con el fin de", "con el objetivo de", "a partir de", "así como", "es decir", "de esta manera",
    "de este modo", "posteriormente", "adicionalmente", "en general", "dado que", "debido a", "ya que",
    "puesto que", "mientras que", "con base en", "respecto a",
]

# Raíces que las guías de estilo señalan como relleno frecuente en texto de modelos de lenguaje.
RAICES_MARCADORAS = [
    "robust", "exhaustiv", "integral", "crucial", "fundamental", "clave", "significativ", "relevan",
    "innovador", "novedos", "eficaz", "eficien", "permit", "abord", "destac", "evidenci", "contribu",
    "aport", "proporcion", "ofrec", "enfoque", "marco", "herramienta", "rigur", "sólid", "comprens",
    "profund", "potencia", "ámbito", "complej",
]

# Raíces marcadoras que además son vocabulario técnico de estadística y
# matemáticas («estimador robusto», «integral», «potencia de la prueba»,
# «aprendizaje profundo», «proporción»). El detector las usa todas; el análisis de vocabulario
# reporta también la serie sin ellas.
RAICES_CON_USO_TECNICO = ["robust", "integral", "potencia", "eficien", "marco", "fundamental", "complej", "significativ", "profund", "proporcion"]

# Tesis de aprendizaje automático, por lo que dicen su título y su resumen.
TESIS_DE_APRENDIZAJE_AUTOMATICO = re.compile(
    r"aprendizaje (?:autom[aá]tico|de m[aá]quina|profundo|supervisado|no supervisado|por refuerzo)|machine learning|"
    r"deep learning|red(?:es)? neuronal|bosques? aleatorios?|random forest|m[aá]quinas? de (?:soporte|vectores)|"
    r"gradient boosting|xgboost|inteligencia artificial",
    re.IGNORECASE,
)

TERMINACIONES = {
    "adverbio_mente": r"\w{4,}mente",
    "nominal_cion": r"\w{3,}(?:ción|ciones|sión|siones)",
    "gerundio": r"\w{2,}(?:ando|iendo|yendo)",
    "primera_plural": r"\w{2,}(?:amos|emos|imos)",
    "participio": r"\w{2,}(?:ado|ada|ados|adas|ido|ida|idos|idas)",
}

VENTANA_MATTR = 50


def limpiar(texto):
    """Deja el texto en una forma común, venga de un PDF o de un modelo: sin
    ligaduras, sin marcas de Markdown ni de LaTeX y con espacios simples."""
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.translate(str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "–": "-", "—": "-"}))
    texto = re.sub(r"\\[a-zA-Z]+|[*_#$`{}^~|\\]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def riqueza_lexica(palabras):
    """Razón tipo-token promediada en ventanas móviles (MATTR)."""
    if len(palabras) < VENTANA_MATTR:
        return len(set(palabras)) / len(palabras)
    ventanas = range(len(palabras) - VENTANA_MATTR + 1)
    return float(np.mean([len(set(palabras[i:i + VENTANA_MATTR])) for i in ventanas])) / VENTANA_MATTR


def variables_de_texto(texto):
    texto = limpiar(texto)
    minusculas = texto.lower()
    palabras = re.findall(r"[a-záéíóúüñ]+", minusculas)
    assert palabras, "texto sin palabras"
    por_cien = 100 / len(palabras)
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", texto) if re.search(r"[a-záéíóúñ]", o.lower())]
    largos_oracion = np.array([len(re.findall(r"[a-záéíóúüñ]+", o.lower())) for o in oraciones])
    largos_palabra = np.array([len(p) for p in palabras])
    fichas = texto.split()

    fila = {
        "oracion_media": largos_oracion.mean(),
        "oracion_cv": largos_oracion.std() / largos_oracion.mean(),
        "palabra_media": largos_palabra.mean(),
        "palabras_largas": (largos_palabra >= 10).mean() * 100,
        "riqueza_lexica": riqueza_lexica(palabras),
        "comas": texto.count(",") * por_cien,
        "punto_y_coma": texto.count(";") * por_cien,
        "dos_puntos": texto.count(":") * por_cien,
        "parentesis": texto.count("(") * por_cien,
        "cifras": sum(any(c.isdigit() for c in f) for f in fichas) * por_cien,
        "siglas": sum(bool(re.fullmatch(r"\W*[A-ZÁÉÍÓÚÑ]{2,}\W*", f)) for f in fichas) * por_cien,
    }
    conteo = pd.Series(palabras).value_counts()
    for palabra in PALABRAS_FUNCION:
        fila[f"f_{palabra}"] = conteo.get(palabra, 0) * por_cien
    for conector in CONECTORES:
        fila[f"c_{conector.replace(' ', '_')}"] = len(re.findall(rf"\b{conector}\b", minusculas)) * por_cien
    for raiz in RAICES_MARCADORAS:
        fila[f"m_{raiz}"] = sum(p.startswith(raiz) for p in palabras) * por_cien
    for nombre, patron in TERMINACIONES.items():
        fila[f"t_{nombre}"] = sum(bool(re.fullmatch(patron, p)) for p in palabras) * por_cien
    return fila


def tabla_de_variables(textos):
    return pd.DataFrame([variables_de_texto(t) for t in textos], index=textos.index)


if __name__ == "__main__":
    ejemplo = variables_de_texto(
        "En este trabajo se propone un modelo robusto. Además, se evalúa con datos de 2019 (n = 120); "
        "finalmente comparamos los resultados."
    )
    assert ejemplo["f_se"] > 0 and ejemplo["c_además"] > 0 and ejemplo["m_robust"] > 0
    assert ejemplo["t_primera_plural"] > 0 and ejemplo["parentesis"] > 0 and ejemplo["punto_y_coma"] > 0
    assert len(ejemplo) == 11 + len(PALABRAS_FUNCION) + len(CONECTORES) + len(RAICES_MARCADORAS) + len(TERMINACIONES)
    print(f"{len(ejemplo)} variables; autochequeo correcto")
