"""Mide cuánto cambian los temas entre dos ventanas de años: entrena un
perceptrón multicapa y una regresión logística para adivinar, solo por el
contenido del resumen, a cuál de las dos ventanas pertenece una tesis.

Un AUC de 0,5 dice que el contenido no distingue las ventanas. La comparación
entre dos ventanas anteriores a la IA da la deriva ordinaria de referencia. Se
usan cuatro representaciones de cada tesis:
  todas las palabras       TF-IDF y LSA
  sin palabras de estilo   lo mismo sin conectores ni raíces marcadoras
  incrustaciones           vector de un modelo de lenguaje multilingüe
  composición              solo área, nivel, sede y longitud del resumen

Todo lo que se ajusta a los datos (TF-IDF, LSA, escalas) va dentro del pliegue.

Lee:     corpus/datos/tesis.csv
Escribe: resultados/clasificacion_periodo.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from corpus import COMPARACIONES, SEMILLA, cargar_tesis, reductor_lsa
from representaciones import incrustaciones

PLIEGUES = 5
REPETICIONES = 3
REPLICAS = 2000


def clasificadores():
    interna = StratifiedKFold(3, shuffle=True, random_state=SEMILLA)
    return {
        "logistica": GridSearchCV(LogisticRegression(max_iter=5000), {"C": [0.1, 1.0, 10.0]}, scoring="neg_log_loss", cv=interna),
        "perceptron": GridSearchCV(
            MLPClassifier(solver="lbfgs", max_iter=2000, random_state=SEMILLA),
            {"hidden_layer_sizes": [(8,), (16,)], "alpha": [1.0, 10.0, 100.0]},
            scoring="neg_log_loss", cv=interna,
        ),
    }


def representaciones(columnas_incrustacion):
    """Transformación de la tabla de tesis en la matriz que recibe el clasificador."""
    texto = lambda quitar_estilo: ColumnTransformer([("lsa", reductor_lsa(quitar_estilo), "documento")])
    return {
        "todas las palabras": texto(False),
        "sin palabras de estilo": texto(True),
        "incrustaciones": ColumnTransformer([("e5", StandardScaler(), columnas_incrustacion)]),
        "composición": ColumnTransformer([
            ("categorias", OneHotEncoder(handle_unknown="ignore"), ["area", "nivel", "sede"]),
            ("longitud", StandardScaler(), ["log_palabras"]),
        ]),
    }


def probabilidad_fuera_de_pliegue(modelo, datos, etiqueta):
    """Promedio, sobre las repeticiones, de la probabilidad que cada tesis recibe
    cuando no entró al entrenamiento."""
    particion = RepeatedStratifiedKFold(n_splits=PLIEGUES, n_repeats=REPETICIONES, random_state=SEMILLA)
    suma = np.zeros(len(datos))
    for entrena, prueba in particion.split(datos, etiqueta):
        ajustado = modelo.fit(datos.iloc[entrena], etiqueta[entrena])
        suma[prueba] += ajustado.predict_proba(datos.iloc[prueba])[:, 1]
    return suma / REPETICIONES


def replicas_de_auc(etiqueta, probabilidad, azar):
    """AUC en remuestreos de tesis. No incluye la variación de volver a entrenar."""
    replicas = []
    while len(replicas) < REPLICAS:
        filas = azar.integers(len(etiqueta), size=len(etiqueta))
        if 0 < etiqueta[filas].mean() < 1:
            replicas.append(roc_auc_score(etiqueta[filas], probabilidad[filas]))
    return np.array(replicas)


def main(tabla_tesis, destino):
    tesis = cargar_tesis(tabla_tesis)
    tesis["log_palabras"] = np.log(tesis["n_palabras_resumen"])
    tesis["sede"] = tesis["sede"].fillna("")
    vectores = incrustaciones(tesis["documento"])
    tesis = pd.concat([tesis, vectores], axis=1)
    azar = np.random.default_rng(SEMILLA)

    filas, replicas = [], {}
    for representacion, transformacion in representaciones(list(vectores.columns)).items():
        for nombre, clasificador in clasificadores().items():
            if representacion == "composición" and nombre != "logistica":
                continue
            for comparacion, (ventana_a, ventana_b) in COMPARACIONES.items():
                en_juego = tesis[tesis["ventana"].isin([ventana_a, ventana_b])].reset_index(drop=True)
                etiqueta = (en_juego["ventana"] == ventana_b).to_numpy().astype(int)
                probabilidad = probabilidad_fuera_de_pliegue(make_pipeline(transformacion, clasificador), en_juego, etiqueta)
                remuestreos = replicas_de_auc(etiqueta, probabilidad, azar)
                replicas[representacion, nombre, comparacion] = remuestreos
                filas.append({
                    "representacion": representacion, "clasificador": nombre, "comparacion": comparacion,
                    "ventana_a": ventana_a, "ventana_b": ventana_b,
                    "n_a": int((etiqueta == 0).sum()), "n_b": int((etiqueta == 1).sum()),
                    "auc": roc_auc_score(etiqueta, probabilidad),
                    "auc_inf": np.quantile(remuestreos, 0.025), "auc_sup": np.quantile(remuestreos, 0.975),
                })
            # Las dos comparaciones comparten la ventana del medio, así que la
            # diferencia se remuestrea con cada una por separado: el intervalo
            # es algo más ancho que con un remuestreo conjunto.
            diferencia = replicas[representacion, nombre, "con IA de por medio"] - replicas[representacion, nombre, "sin IA de por medio"]
            filas.append({
                "representacion": representacion, "clasificador": nombre, "comparacion": "diferencia",
                "auc": filas[-1]["auc"] - filas[-2]["auc"],
                "auc_inf": np.quantile(diferencia, 0.025), "auc_sup": np.quantile(diferencia, 0.975),
            })
        print(f"{representacion}: lista", flush=True)
    pd.DataFrame(filas).to_csv(destino, index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
