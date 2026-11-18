"""Forma de la nube de tesis y de la nube de palabras, por ventana de años:
escala, homología persistente y contraste por permutación.

Nubes:
  tesis (LSA)             cada tesis es su vector LSA
  tesis (incrustaciones)  cada tesis es su vector de un modelo de lenguaje
  palabras                cada término es su perfil de uso en las tesis de la ventana

En cada nube se construye el complejo de Vietoris-Rips. Dos ventanas se comparan
con siete estadísticos:
  diferencia de distancia media  la distancia media entre dos puntos es la escala
                                 de la nube: baja cuando las tesis se parecen más
  diferencia de L0               L0 es la suma de las vidas de las componentes, que
                                 es la longitud del árbol generador mínimo: mide la
                                 distancia entre vecinos cercanos
  diferencia de L1               L1 es la suma de las vidas de los ciclos
  W0 y W1                        distancia de Wasserstein de orden 1 entre los
                                 diagramas de las dos ventanas
  W0 y W1 a escala               las mismas distancias después de dividir cada nube
                                 por su distancia media, para separar el cambio de
                                 forma del cambio de escala

Todas las nubes que se comparan tienen el mismo número de tesis, porque la
persistencia total crece con el tamaño de la muestra. Las permutaciones barajan
la ventana dentro de cada estrato de área y sede.

Lee:  corpus/datos/tesis.csv
Escribe en resultados/:
  homologia_contrastes.csv   estadístico observado, intervalo nulo y valor p de las dos comparaciones
  homologia_sensibilidad.csv lo mismo sin las tesis de aprendizaje automático
  homologia_compacidad.csv   distancia media entre las tesis de aprendizaje automático y entre tesis al azar
  homologia_potencia.csv     proporción de rechazos cuando el cambio es conocido
  lsa.csv                    varianza que retiene la reducción LSA
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from persim import wasserstein
from ripser import ripser
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from corpus import COMPARACIONES, SEMILLA, VENTANAS, cargar_tesis, vectores_de_incrustacion, vectores_lsa

TESIS_POR_SUBMUESTRA = 150
PARES_POR_CONTRASTE = 10
PERMUTACIONES = 2000
TERMINOS = 200
COMPONENTES_PALABRAS = 30
ESTADISTICOS = ["diferencia de distancia media", "diferencia de L0", "diferencia de L1", "W0", "W1", "W0 a escala", "W1 a escala"]
# Las diferencias se contrastan a dos colas; las distancias solo rechazan por valores grandes.
A_DOS_COLAS = np.array([True, True, True, False, False, False, False])
NUBES_DE_TESIS = ["tesis (LSA)", "tesis (incrustaciones)"]
SORTEOS_DE_COMPACIDAD = 5000
# Prueba con un cambio conocido: en una mitad de la ventana 2018-2021 se
# reemplaza una fracción de las tesis por tesis de aprendizaje automático de 2023-2026.
FRACCIONES_PLANTADAS = [0.0, 0.1, 0.2, 0.3]
REPLICAS_DE_POTENCIA = 40
PERMUTACIONES_DE_POTENCIA = 99
PARES_DE_POTENCIA = 5
NIVEL = 0.05


def resumen_de(nube):
    """Distancia media entre puntos, vidas de las componentes (sin la que nunca
    muere) y diagrama de los ciclos."""
    h0, h1 = ripser(nube, maxdim=1)["dgms"]
    vidas = np.sort(h0[np.isfinite(h0[:, 1]), 1])
    # Dos puntos idénticos dan una vida de longitud cero que ripser omite: se repone.
    return pdist(nube).mean(), np.pad(vidas, (len(nube) - 1 - len(vidas), 0)), h1


def comparar(nube_a, nube_b):
    """Los siete estadísticos entre dos nubes del mismo tamaño, y la distancia
    media, L0 y L1 de la primera, que sirven de nivel de referencia."""
    escala_a, vidas_a, ciclos_a = resumen_de(nube_a)
    escala_b, vidas_b, ciclos_b = resumen_de(nube_b)
    l1_a, l1_b = (ciclos_a[:, 1] - ciclos_a[:, 0]).sum(), (ciclos_b[:, 1] - ciclos_b[:, 0]).sum()
    # En H0 todas las componentes nacen en cero: la distancia de Wasserstein
    # entre los dos diagramas es la suma de diferencias entre vidas ordenadas.
    estadisticos = [
        escala_b - escala_a,
        vidas_b.sum() - vidas_a.sum(),
        l1_b - l1_a,
        np.abs(vidas_b - vidas_a).sum(),
        wasserstein(ciclos_a, ciclos_b, matching=False),
        np.abs(vidas_b / escala_b - vidas_a / escala_a).sum(),
        wasserstein(ciclos_a / escala_a, ciclos_b / escala_b, matching=False),
    ]
    return np.array(estadisticos), np.array([escala_a, vidas_a.sum(), l1_a])


class Nubes:
    """Construye la nube de puntos de un conjunto de tesis, según el tipo."""

    def __init__(self, tesis):
        self.tfidf, _, self.lsa, self.varianza_lsa = vectores_lsa(tesis["documento"])
        self.tfidf = self.tfidf.tocsr()
        self.incrustacion = vectores_de_incrustacion(tesis["documento"])
        en_ventanas = np.flatnonzero(tesis["ventana"] != "")
        frecuencia = np.asarray((self.tfidf[en_ventanas] > 0).sum(axis=0)).ravel()
        self.terminos = np.argsort(frecuencia)[::-1][:TERMINOS]

    def __call__(self, tipo, filas):
        if tipo == "tesis (LSA)":
            return self.lsa[filas]
        if tipo == "tesis (incrustaciones)":
            return self.incrustacion[filas]
        perfiles = self.tfidf[filas][:, self.terminos].T
        return normalize(TruncatedSVD(COMPONENTES_PALABRAS, random_state=SEMILLA).fit_transform(perfiles))


def estadistico_medio(nube_de, grupo_a, grupo_b, azar, tamano, pares):
    """Media de los estadísticos y de los niveles en varios pares de submuestras,
    una de cada grupo. El observado y cada permutación se calculan con esta función."""
    valores = [
        comparar(nube_de(azar.choice(grupo_a, tamano, replace=False)), nube_de(azar.choice(grupo_b, tamano, replace=False)))
        for _ in range(pares)
    ]
    return np.mean([v[0] for v in valores], axis=0), np.mean([v[1] for v in valores], axis=0)


def barajar_por_estrato(grupo_a, grupo_b, estratos, azar):
    """Reparte al azar las tesis entre los dos grupos sin cambiar cuántas de
    cada estrato tiene cada uno."""
    juntas = np.concatenate([grupo_a, grupo_b])
    es_del_b = np.arange(len(juntas)) >= len(grupo_a)
    for estrato in np.unique(estratos[juntas]):
        del_estrato = np.flatnonzero(estratos[juntas] == estrato)
        es_del_b[del_estrato] = azar.permutation(es_del_b[del_estrato])
    return juntas[~es_del_b], juntas[es_del_b]


def contraste(nube_de, grupo_a, grupo_b, estratos, permutaciones, azar, tamano, pares):
    """Valor p de cada estadístico por permutación. A dos colas se mide la
    distancia al centro de la distribución nula, que no tiene por qué ser cero."""
    observado, niveles = estadistico_medio(nube_de, grupo_a, grupo_b, azar, tamano, pares)
    nulos = np.array([
        estadistico_medio(nube_de, *barajar_por_estrato(grupo_a, grupo_b, estratos, azar), azar, tamano, pares)[0]
        for _ in range(permutaciones)
    ])
    centro = nulos.mean(axis=0)
    extremos = np.where(A_DOS_COLAS, np.abs(nulos - centro) >= np.abs(observado - centro), nulos >= observado).sum(axis=0)
    return observado, niveles, nulos, (extremos + 1) / (permutaciones + 1)


def filas_del_contraste(nubes, tipo, comparacion, ventanas, grupo_a, grupo_b, estratos, semilla):
    azar = np.random.default_rng(semilla)
    observado, niveles, nulos, valores_p = contraste(
        lambda filas: nubes(tipo, filas), grupo_a, grupo_b, estratos, PERMUTACIONES, azar, TESIS_POR_SUBMUESTRA, PARES_POR_CONTRASTE
    )
    return [{
        "nube": tipo, "comparacion": comparacion, "ventana_a": ventanas[0], "ventana_b": ventanas[1],
        "estadistico": estadistico, "observado": observado[i],
        # El nivel de la primera ventana solo existe para las tres diferencias.
        "nivel_de_la_primera": niveles[i] if i < len(niveles) else np.nan,
        "nula_inf": np.quantile(nulos[:, i], 0.025), "nula_sup": np.quantile(nulos[:, i], 0.975),
        "valor_p": valores_p[i],
    } for i, estadistico in enumerate(ESTADISTICOS)]


def compacidad(nube, del_tema, de_la_ventana, azar):
    """¿Están las tesis de un tema más juntas que un conjunto al azar de la
    misma ventana y del mismo tamaño?"""
    del_tema_media = pdist(nube[del_tema]).mean()
    al_azar = np.array([pdist(nube[azar.choice(de_la_ventana, len(del_tema), replace=False)]).mean() for _ in range(SORTEOS_DE_COMPACIDAD)])
    return {
        "tesis": len(del_tema), "distancia_media": del_tema_media, "distancia_media_al_azar": al_azar.mean(),
        "valor_p": ((al_azar <= del_tema_media).sum() + 1) / (SORTEOS_DE_COMPACIDAD + 1),
    }


def valores_p_con_cambio_plantado(nubes, de_la_ventana, plantables, estratos, fraccion, tamano, semilla):
    """Parte la ventana en dos mitades al azar, entre las que no hay diferencia
    real, cambia una fracción de la segunda por tesis del tema plantado y
    contrasta las dos mitades. El grupo se planta antes de barajar."""
    azar = np.random.default_rng(semilla)
    mitades = azar.permutation(de_la_ventana)
    mitad_a, mitad_b = mitades[:len(mitades) // 2], mitades[len(mitades) // 2:]
    plantadas = int(round(fraccion * len(mitad_b)))
    mitad_b = np.concatenate([mitad_b[plantadas:], azar.choice(plantables, plantadas, replace=False)])
    return contraste(
        lambda filas: nubes("tesis (LSA)", filas), mitad_a, mitad_b, estratos, PERMUTACIONES_DE_POTENCIA, azar, tamano, PARES_DE_POTENCIA
    )[3]


def main(tabla_tesis, carpeta):
    tesis = cargar_tesis(tabla_tesis)
    nubes = Nubes(tesis)
    estratos = tesis["estrato"].to_numpy()
    de_aprendizaje = tesis["es_de_aprendizaje_automatico"].to_numpy()
    filas_de = {v: np.flatnonzero(tesis["ventana"] == v) for v in VENTANAS}
    sin_aprendizaje = {v: filas[~de_aprendizaje[filas]] for v, filas in filas_de.items()}
    assert min(len(f) for f in sin_aprendizaje.values()) >= TESIS_POR_SUBMUESTRA

    con_ia = COMPARACIONES["con IA de por medio"]
    tareas = [(tipo, comparacion, ventanas, filas_de) for tipo in [*NUBES_DE_TESIS, "palabras"] for comparacion, ventanas in COMPARACIONES.items()]
    tareas += [(tipo, "con IA de por medio, sin aprendizaje automático", con_ia, sin_aprendizaje) for tipo in NUBES_DE_TESIS]
    # Cada tarea tiene su propio generador: el resultado no depende del orden en que terminen.
    semillas = np.random.SeedSequence(SEMILLA).spawn(len(tareas) + len(FRACCIONES_PLANTADAS) * REPLICAS_DE_POTENCIA + 1)
    resultados = Parallel(n_jobs=-2)(
        delayed(filas_del_contraste)(nubes, tipo, comparacion, ventanas, grupos[ventanas[0]], grupos[ventanas[1]], estratos, semilla)
        for (tipo, comparacion, ventanas, grupos), semilla in zip(tareas, semillas)
    )
    contrastes = pd.DataFrame([fila for filas in resultados for fila in filas])
    principal = contrastes["comparacion"].isin(COMPARACIONES)
    contrastes[principal].to_csv(carpeta / "homologia_contrastes.csv", index=False)
    contrastes[~principal].to_csv(carpeta / "homologia_sensibilidad.csv", index=False)
    print("contrastes: listos", flush=True)

    recientes = filas_de[con_ia[1]]
    del_tema = recientes[de_aprendizaje[recientes]]
    azar = np.random.default_rng(semillas[-1])
    pd.DataFrame([
        {"nube": tipo, **compacidad(nubes(tipo, np.arange(len(tesis))), del_tema, recientes, azar)} for tipo in NUBES_DE_TESIS
    ]).to_csv(carpeta / "homologia_compacidad.csv", index=False)

    anteriores = filas_de[con_ia[0]]
    tamano = len(anteriores) // 2 - 10
    assert int(round(max(FRACCIONES_PLANTADAS) * (len(anteriores) - len(anteriores) // 2))) <= len(del_tema)
    casos = [(fraccion, replica) for fraccion in FRACCIONES_PLANTADAS for replica in range(REPLICAS_DE_POTENCIA)]
    valores_p = Parallel(n_jobs=-2)(
        delayed(valores_p_con_cambio_plantado)(nubes, anteriores, del_tema, estratos, fraccion, tamano, semilla)
        for (fraccion, _), semilla in zip(casos, semillas[len(tareas):])
    )
    potencia = pd.DataFrame(valores_p, columns=ESTADISTICOS).assign(fraccion_plantada=[fraccion for fraccion, _ in casos])
    potencia = potencia.melt(id_vars="fraccion_plantada", var_name="estadistico", value_name="valor_p")
    potencia.groupby(["fraccion_plantada", "estadistico"], as_index=False, sort=False).agg(
        replicas=("valor_p", "size"), rechazos=("valor_p", lambda p: int((p < NIVEL).sum()))
    ).assign(tesis_por_nube=tamano).to_csv(carpeta / "homologia_potencia.csv", index=False)
    pd.DataFrame([{"clave": "varianza_retenida_lsa", "valor": nubes.varianza_lsa}]).to_csv(carpeta / "lsa.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
