"""Escoge al azar párrafos de prosa del cuerpo de cada tesis.

Lee:     datos/texto.nosync/<uuid>.txt (salida de extraer_texto.py)
Escribe: datos/parrafos_cuerpo.csv con uuid, orden, texto y n_palabras.

Un PDF pasado a texto mezcla índice, encabezados, tablas, fórmulas y pies de
página. Solo se aceptan bloques que empiezan en mayúscula, terminan en punto y
están hechos casi por completo de palabras, con la proporción de palabras de
función que tiene la prosa en español.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from texto_pdf import reparar

SEMILLA = 20261118
PARRAFOS_POR_TESIS = 4
MIN_PALABRAS, MAX_PALABRAS = 60, 250
MIN_PALABRAS_BLOQUE = 40
MIN_FRACCION_FUNCION = 0.30
MIN_FRACCION_ALFABETICA = 0.85
PALABRAS_FUNCION = set(
    "de la que el en y a los del se las por un para con no una su al lo como más o es son este esta entre".split()
)
PALABRA_SUELTA = re.compile(r"[^\W\d_]+[.,;:)]?|\([^\W\d_]+")
NO_ES_CUERPO = re.compile(r"agrade|dedicatoria|derechos reservados|palabras clave|keywords", re.IGNORECASE)


def unir_lineas(bloque):
    bloque = re.sub(r"(\w)-\n(\w)", r"\1\2", bloque)
    return re.sub(r"\s+", " ", bloque).strip()


def es_texto_corrido(fichas):
    alfabeticas = sum(bool(PALABRA_SUELTA.fullmatch(f)) for f in fichas)
    de_funcion = sum(f.lower() in PALABRAS_FUNCION for f in fichas)
    return alfabeticas / len(fichas) >= MIN_FRACCION_ALFABETICA and de_funcion / len(fichas) >= MIN_FRACCION_FUNCION


def es_prosa(bloque):
    fichas = bloque.split()
    if not MIN_PALABRAS <= len(fichas) <= MAX_PALABRAS:
        return False
    if not (bloque[0].isupper() and bloque.endswith(".")) or NO_ES_CUERPO.search(bloque):
        return False
    return es_texto_corrido(fichas)


def bloques_de_texto(texto):
    """Bloques de texto corrido de cualquier longitud. Sirven para contar
    palabras; para el detector se usan solo los que además son un párrafo entero."""
    bloques = (unir_lineas(b) for b in re.split(r"\n\s*\n", reparar(texto)))
    return [b for b in bloques if len(b.split()) >= MIN_PALABRAS_BLOQUE and es_texto_corrido(b.split())]


def parrafos_de_prosa(texto):
    return [b for b in bloques_de_texto(texto) if es_prosa(b)]


def main(carpeta_texto, destino):
    filas = []
    archivos = sorted(carpeta_texto.glob("*.txt"))
    for archivo in archivos:
        prosa = parrafos_de_prosa(archivo.read_text(encoding="utf-8", errors="replace"))
        if len(prosa) < PARRAFOS_POR_TESIS:
            continue
        # Semilla propia de cada tesis: su muestra no cambia cuando se bajan más tesis.
        azar = np.random.default_rng([SEMILLA, int(archivo.stem[:8], 16)])
        elegidos = sorted(azar.choice(len(prosa), PARRAFOS_POR_TESIS, replace=False))
        for orden, indice in enumerate(elegidos, start=1):
            filas.append({"uuid": archivo.stem, "orden": orden, "texto": prosa[indice], "n_palabras": len(prosa[indice].split())})
    parrafos = pd.DataFrame(filas)
    parrafos.to_csv(destino, index=False)
    print(f"{len(archivos)} tesis con texto; {parrafos['uuid'].nunique()} con al menos {PARRAFOS_POR_TESIS} párrafos de prosa")


if __name__ == "__main__":
    prosa = (
        "La estimación de los parámetros de un modelo es uno de los problemas que más se estudian en la "
        "estadística y en este trabajo se aborda con datos de la región, para lo cual se usa un modelo que "
        "permite comparar los resultados con los de otros autores y con los que se obtienen por otra vía, "
        "de modo que el lector pueda juzgar la calidad del ajuste en cada uno de los casos que se presentan."
    )
    assert es_prosa(prosa)
    assert not es_prosa("Tabla 2.1: 62534 3667 491 75 " * 12 + ".")
    assert not es_prosa("Agradezco a mis padres. " + prosa)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
