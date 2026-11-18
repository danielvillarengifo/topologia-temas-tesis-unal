"""Carga las tesis y las representa como vectores de temas. Lo usan los demás
scripts de esta carpeta."""
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize

from representaciones import incrustaciones
from variables import CONECTORES, RAICES_CON_USO_TECNICO, RAICES_MARCADORAS, TESIS_DE_APRENDIZAJE_AUTOMATICO

SEMILLA = 20261118
DIMENSIONES = 50

# Ventanas de cuatro años. Las dos primeras son anteriores a la IA generativa:
# compararlas mide cuánto cambia el corpus por deriva ordinaria.
VENTANAS = {"2014-2017": (2014, 2017), "2018-2021": (2018, 2021), "2023-2026": (2023, 2026)}
COMPARACIONES = {
    "sin IA de por medio": ("2014-2017", "2018-2021"),
    "con IA de por medio": ("2018-2021", "2023-2026"),
}

PALABRAS_VACIAS = sorted(set(
    "de la que el en y a los del se las por un para con no una su al lo como más o pero sus le ya este esta "
    "entre cuando muy sin sobre también hasta donde desde todo nos durante uno ni otros ese eso ante e esto "
    "antes algunos unos otro otras otra tanto esa estos mucho cual poco estas algunas algo son es fue ser ha "
    "han sido está están puede pueden cada dos así bajo dicho dicha tal tales cuyo cuya mediante según tras "
    "hacia aquí si sea sean tiene tienen hace hacen fueron era eran hay cuales además embargo sino aunque les "
    "the of and in to is for this that with are we on an as by be from which".split()
))

# Las frases van primero y de la más larga a la más corta, para quitar «por lo
# tanto» entero y no solo «tanto».
ESTILO = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, CONECTORES), key=len, reverse=True)) + r")\b|\b(?:"
    + "|".join(r for r in RAICES_MARCADORAS if r not in RAICES_CON_USO_TECNICO) + r")\w*",
    re.IGNORECASE,
)


def cargar_tesis(tabla):
    tesis = pd.read_csv(tabla)
    tesis["documento"] = tesis["titulo"] + ". " + tesis["resumen"]
    tesis["ventana"] = ""
    for nombre, (desde, hasta) in VENTANAS.items():
        tesis.loc[tesis["anio"].between(desde, hasta), "ventana"] = nombre
    tesis["es_de_aprendizaje_automatico"] = tesis["documento"].str.contains(TESIS_DE_APRENDIZAJE_AUTOMATICO)
    # Estrato de las permutaciones: un cambio en la mezcla de áreas o de sedes
    # no debe pasar por un cambio de temas.
    tesis["estrato"] = tesis["area"] + " / " + tesis["sede"].fillna("")
    return tesis


def sin_estilo(documento):
    """El documento sin los conectores ni las raíces marcadoras de estilo que no
    tienen uso técnico. Lo que queda habla del tema y no de cómo está escrito."""
    return ESTILO.sub(" ", documento)


def vectorizador_tfidf(quitar_estilo=False):
    return TfidfVectorizer(
        lowercase=True,
        preprocessor=(lambda d: sin_estilo(d).lower()) if quitar_estilo else None,
        token_pattern=r"(?u)\b[a-záéíóúüñ]{3,}\b",
        stop_words=PALABRAS_VACIAS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        sublinear_tf=True,
    )


def reductor_lsa(quitar_estilo=False, dimensiones=DIMENSIONES):
    """TF-IDF, análisis semántico latente y norma uno. Como tubería, para poder
    ajustarla solo con los datos de entrenamiento de cada pliegue."""
    return make_pipeline(vectorizador_tfidf(quitar_estilo), TruncatedSVD(dimensiones, random_state=SEMILLA), Normalizer())


def vectores_lsa(documentos, dimensiones=DIMENSIONES):
    """Matriz TF-IDF, sus términos, los vectores LSA de norma uno y la fracción
    de varianza que retienen. Con norma uno, la distancia euclídea entre dos
    tesis es función de su coseno."""
    vectorizador = vectorizador_tfidf()
    tfidf = vectorizador.fit_transform(documentos)
    svd = TruncatedSVD(dimensiones, random_state=SEMILLA)
    vectores = normalize(svd.fit_transform(tfidf))
    return tfidf, np.array(vectorizador.get_feature_names_out()), vectores, svd.explained_variance_ratio_.sum()


def vectores_de_incrustacion(documentos, dimensiones=DIMENSIONES):
    """Incrustaciones de un modelo de lenguaje multilingüe, reducidas a las
    mismas dimensiones que el LSA y con norma uno."""
    completas = incrustaciones(documentos).to_numpy()
    return normalize(PCA(dimensiones, random_state=SEMILLA).fit_transform(completas))
