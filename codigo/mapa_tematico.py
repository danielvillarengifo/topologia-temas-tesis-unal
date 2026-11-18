"""Organiza las tesis en un mapa autoorganizado de Kohonen y contrasta si las
tesis de una ventana se reparten en el mapa igual que las de la anterior.

El contraste se hace en las dos comparaciones del artículo (con y sin IA de por
medio), en veinte mapas con semillas distintas y con dos representaciones de
las tesis (LSA e incrustaciones de un modelo de lenguaje).

Lee:  corpus/datos/tesis.csv
Escribe en resultados/:
  som_contrastes.csv    por representación, semilla y comparación: calidad del mapa, ji cuadrado y valor p
  som_estabilidad.csv   acuerdo entre los mapas de distintas semillas (índice de Rand ajustado)
  som_unidades.csv      del mapa escogido: posición, tesis por ventana y términos de cada unidad
  som_destacadas.csv    unidades con mayor exceso y defecto de tesis recientes en el mapa escogido
  temas_por_ventana.csv tesis de aprendizaje automático y tesis escritas en inglés por ventana
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from corpus import COMPARACIONES, SEMILLA, VENTANAS, cargar_tesis, vectores_de_incrustacion, vectores_lsa

FILAS, COLUMNAS = 8, 8
EPOCAS = 30
TASA_INICIAL, TASA_FINAL = 0.5, 0.01
RADIO_FINAL = 0.5
MAPAS = 20
PERMUTACIONES = 2000
TERMINOS_POR_UNIDAD = 4
UNIDADES_DESTACADAS = 5


def entrenar_som(vectores, azar):
    """Regla de Kohonen en línea: gana la unidad más cercana y ella y sus vecinas
    de la rejilla se acercan al dato. Tasa y radio decrecen de forma geométrica."""
    rejilla = np.array([(f, c) for f in range(FILAS) for c in range(COLUMNAS)], dtype=float)
    pesos = vectores[azar.choice(len(vectores), len(rejilla), replace=False)].copy()
    radio_inicial = max(FILAS, COLUMNAS) / 2
    pasos = EPOCAS * len(vectores)
    paso = 0
    for _ in range(EPOCAS):
        for vector in vectores[azar.permutation(len(vectores))]:
            avance = paso / pasos
            tasa = TASA_INICIAL * (TASA_FINAL / TASA_INICIAL) ** avance
            radio = radio_inicial * (RADIO_FINAL / radio_inicial) ** avance
            ganadora = np.argmin(((pesos - vector) ** 2).sum(axis=1))
            vecindad = np.exp(-((rejilla - rejilla[ganadora]) ** 2).sum(axis=1) / (2 * radio**2))
            pesos += tasa * vecindad[:, None] * (vector - pesos)
            paso += 1
    return pesos, rejilla


def calidad(vectores, pesos, rejilla):
    distancias = np.linalg.norm(vectores[:, None, :] - pesos[None, :, :], axis=2)
    orden = np.argsort(distancias, axis=1)
    primera, segunda = orden[:, 0], orden[:, 1]
    error_cuantizacion = distancias[np.arange(len(vectores)), primera].mean()
    # Vecinas: las ocho unidades que rodean a la ganadora en la rejilla.
    no_vecinas = np.abs(rejilla[primera] - rejilla[segunda]).max(axis=1) > 1
    return primera, error_cuantizacion, no_vecinas.mean()


def tabla_de_contingencia(unidad, es_reciente):
    tabla = pd.crosstab(unidad, es_reciente).to_numpy()
    esperado = tabla.sum(axis=1, keepdims=True) * tabla.sum(axis=0, keepdims=True) / tabla.sum()
    return tabla, esperado


def ji_cuadrado(unidad, es_reciente):
    tabla, esperado = tabla_de_contingencia(unidad, es_reciente)
    return ((tabla - esperado) ** 2 / esperado).sum()


def contraste(unidad, es_reciente, estrato, azar):
    """Valor p por permutación de la asociación entre unidad y ventana. La
    ventana se permuta dentro de cada estrato de área y sede."""
    observado = ji_cuadrado(unidad, es_reciente)
    filas_por_estrato = [np.flatnonzero(estrato == e) for e in np.unique(estrato)]
    mayores = 0
    for _ in range(PERMUTACIONES):
        permutado = es_reciente.copy()
        for filas in filas_por_estrato:
            permutado[filas] = es_reciente[azar.permutation(filas)]
        mayores += ji_cuadrado(unidad, permutado) >= observado
    return observado, (mayores + 1) / (PERMUTACIONES + 1)


def describir_unidades(tesis, unidad, rejilla, tfidf, terminos):
    filas = []
    for u in range(len(rejilla)):
        miembros = np.flatnonzero(unidad == u)
        peso_medio = np.asarray(tfidf[miembros].mean(axis=0)).ravel() if len(miembros) else np.zeros(len(terminos))
        fila = {"unidad": u, "fila": int(rejilla[u, 0]), "columna": int(rejilla[u, 1])}
        for ventana in VENTANAS:
            fila[ventana] = int((tesis["ventana"].to_numpy()[miembros] == ventana).sum())
        fila["terminos"] = ", ".join(terminos[np.argsort(peso_medio)[::-1][:TERMINOS_POR_UNIDAD]]) if len(miembros) else ""
        filas.append(fila)
    return pd.DataFrame(filas)


def destacar(unidades):
    """Residuo de Pearson de cada unidad en la comparación con IA de por medio."""
    anterior, reciente = COMPARACIONES["con IA de por medio"]
    total = unidades[anterior] + unidades[reciente]
    esperado = total * unidades[reciente].sum() / total.sum()
    unidades = unidades.assign(residuo=(unidades[reciente] - esperado) / np.sqrt(esperado.where(esperado > 0)))
    extremos = pd.concat([
        unidades.nlargest(UNIDADES_DESTACADAS, "residuo").assign(sentido="Crecen"),
        unidades.nsmallest(UNIDADES_DESTACADAS, "residuo").assign(sentido="Se reducen"),
    ]).sort_values("residuo", ascending=False)
    return unidades, extremos.assign(letra=list("ABCDEFGHIJ")[:len(extremos)])


def main(tabla_tesis, carpeta):
    tesis = cargar_tesis(tabla_tesis)
    tfidf, terminos, lsa, _ = vectores_lsa(tesis["documento"])
    nubes = {"LSA": lsa, "incrustaciones": vectores_de_incrustacion(tesis["documento"])}

    contrastes, estabilidad, mapas_lsa = [], [], []
    for representacion, vectores in nubes.items():
        asignaciones = []
        for semilla in range(SEMILLA, SEMILLA + MAPAS):
            azar = np.random.default_rng(semilla)
            pesos, rejilla = entrenar_som(vectores, azar)
            unidad, error_cuantizacion, error_topografico = calidad(vectores, pesos, rejilla)
            asignaciones.append(unidad)
            if representacion == "LSA":
                mapas_lsa.append((error_cuantizacion, unidad, rejilla))
            for comparacion, (ventana_a, ventana_b) in COMPARACIONES.items():
                en_juego = tesis["ventana"].isin([ventana_a, ventana_b]).to_numpy()
                es_reciente = (tesis["ventana"] == ventana_b).to_numpy()[en_juego]
                estadistico, valor_p = contraste(unidad[en_juego], es_reciente, tesis["estrato"].to_numpy()[en_juego], azar)
                contrastes.append({
                    "representacion": representacion, "semilla": semilla, "comparacion": comparacion,
                    "error_cuantizacion": error_cuantizacion, "error_topografico": error_topografico,
                    "ji_cuadrado": estadistico, "valor_p": valor_p,
                })
        acuerdos = [adjusted_rand_score(a, b) for a, b in combinations(asignaciones, 2)]
        estabilidad.append({"representacion": representacion, "rand_ajustado_medio": np.mean(acuerdos), "rand_ajustado_min": np.min(acuerdos)})
        print(f"mapas con {representacion}: listos", flush=True)

    _, unidad, rejilla = min(mapas_lsa, key=lambda m: m[0])
    unidades, destacadas = destacar(describir_unidades(tesis, unidad, rejilla, tfidf, terminos))
    pd.DataFrame(contrastes).to_csv(carpeta / "som_contrastes.csv", index=False)
    pd.DataFrame(estabilidad).to_csv(carpeta / "som_estabilidad.csv", index=False)
    unidades.to_csv(carpeta / "som_unidades.csv", index=False)
    destacadas.to_csv(carpeta / "som_destacadas.csv", index=False)
    tesis[tesis["ventana"] != ""].groupby("ventana", as_index=False).agg(
        tesis=("uuid", "size"), de_aprendizaje_automatico=("es_de_aprendizaje_automatico", "sum"),
        en_ingles=("idioma", lambda idioma: int((idioma == "eng").sum())),
    ).to_csv(carpeta / "temas_por_ventana.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
