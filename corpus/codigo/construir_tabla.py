"""Convierte metadatos_crudos.jsonl en dos tablas con una fila por tesis (año,
nivel, área, programa y resumen en español normalizado): tesis.csv, que es el
conjunto de análisis, y tesis_excluidas.csv, con las de enseñanza."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

from texto_pdf import limpiar_resumen, reparar

MIN_PALABRAS_RESUMEN = 60
MIN_MENCIONES_EN_RESUMEN = 2
# El Departamento de Matemáticas y Estadística de Manizales también aloja la
# Maestría en Enseñanza de las Ciencias Exactas y Naturales. Son tesis de
# didáctica, concentradas entre 2011 y 2021, y no de estadística o matemáticas.
PATRON_ENSENANZA = re.compile(
    r"enseñanza|didáctic|\baulas?\b|pedagóg|estudiantes de grado|secundaria|"
    r"aprendizaje significativo|educación básica|educación media"
)
CAMPOS_DE_PROGRAMA = ["dc.publisher.program", "dc.description.degreename"]
CAMPOS_DE_DEPARTAMENTO = ["dc.relation.ispartof", "dc.publisher.department"]
DEPARTAMENTO_MIXTO = "Departamento de Matemáticas y Estadística"
SEDES = ["Bogotá", "Medellín", "Manizales", "Palmira"]


def normalizar(texto):
    """Quita lo que delata el origen del texto y no su estilo: saltos de línea,
    guiones de corte de PDF, comillas tipográficas y espacios raros."""
    texto = reparar(texto)
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)
    texto = texto.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", texto).strip()


def valores(metadata, campo):
    return [v["value"] for v in metadata.get(campo, []) if v.get("value")]


def primero(metadata, campo):
    encontrados = valores(metadata, campo)
    return encontrados[0] if encontrados else ""


def clasificar_area(metadata):
    """El programa o el título mandan; el departamento solo decide si ellos no
    dicen nada. El departamento de Manizales nombra las dos áreas: sin más
    pistas la tesis es de Matemáticas, que es el área de sus posgrados de
    investigación y como quedan catalogadas sus tesis desde 2022."""
    for campos in (CAMPOS_DE_PROGRAMA, CAMPOS_DE_DEPARTAMENTO):
        texto = " ".join(v for campo in campos for v in valores(metadata, campo))
        sin_mixto = texto.replace(DEPARTAMENTO_MIXTO, "")
        if "Estadística" in sin_mixto:
            return "Estadística"
        if "Matemática" in sin_mixto:
            return "Matemáticas"
        if DEPARTAMENTO_MIXTO in texto:
            return "Matemáticas"
    return ""


def clasificar_sede(metadata):
    texto = " ".join(v for campo in [*CAMPOS_DE_PROGRAMA, *CAMPOS_DE_DEPARTAMENTO, "dc.publisher.branch"] for v in valores(metadata, campo))
    return next((sede for sede in SEDES if sede in texto), "")


def es_de_ensenanza(programa, titulo, palabras_clave, resumen):
    """Una mención suelta en el resumen no basta: una tesis de estadística puede
    analizar datos de educación. Dos o más sí."""
    return (
        "Enseñanza" in programa
        or bool(PATRON_ENSENANZA.search(f"{titulo} {palabras_clave}".lower()))
        or len(PATRON_ENSENANZA.findall(resumen.lower())) >= MIN_MENCIONES_EN_RESUMEN
    )


def fila(item):
    metadata = item["metadata"]
    resumenes = [limpiar_resumen(normalizar(r)) for r in valores(metadata, "dc.description.abstract")]
    resumen = max(resumenes, key=len, default="")
    fecha = primero(metadata, "dc.date.issued")
    titulo = normalizar(primero(metadata, "dc.title"))
    palabras_clave = "; ".join(dict.fromkeys(normalizar(p) for p in valores(metadata, "dc.subject.proposal")))
    programa = primero(metadata, "dc.publisher.program") or primero(metadata, "dc.relation.ispartof")
    return {
        "uuid": item["uuid"],
        "handle": item["handle"],
        "anio": int(fecha[:4]) if fecha[:4].isdigit() else None,
        "nivel": "Doctorado" if "Doctorado" in primero(metadata, "dc.type") else "Maestría",
        "area": clasificar_area(metadata),
        "programa": programa,
        "es_ensenanza": es_de_ensenanza(programa, titulo, palabras_clave, resumen),
        "sede": clasificar_sede(metadata),
        "idioma": primero(metadata, "dc.language.iso"),
        "titulo": titulo,
        "autor": primero(metadata, "dc.contributor.author"),
        "director": "; ".join(valores(metadata, "dc.contributor.advisor")),
        "palabras_clave": palabras_clave,
        "n_referencias": len(valores(metadata, "dc.relation.references")),
        "resumen": resumen,
        "n_palabras_resumen": len(resumen.split()),
    }


def main(origen, destino, destino_excluidas):
    with open(origen, encoding="utf-8") as crudo:
        tesis = pd.DataFrame(fila(json.loads(linea)) for linea in crudo)
    total = len(tesis)
    tesis = tesis[tesis["anio"].notna() & (tesis["area"] != "")]
    tesis = tesis[tesis["n_palabras_resumen"] >= MIN_PALABRAS_RESUMEN]
    tesis = tesis.astype({"anio": int}).sort_values(["anio", "uuid"])
    assert tesis["uuid"].is_unique
    # Algunas tesis están depositadas dos veces, con otro identificador.
    tesis = tesis[~tesis.assign(titulo=tesis["titulo"].str.lower()).duplicated(["titulo", "autor"])]
    excluidas = tesis[tesis["es_ensenanza"]].drop(columns="es_ensenanza")
    analisis = tesis[~tesis["es_ensenanza"]].drop(columns="es_ensenanza")
    analisis.to_csv(destino, index=False)
    excluidas.to_csv(destino_excluidas, index=False)
    print(
        f"{total} tesis bajadas; {len(tesis)} con año, área y resumen en español de al menos "
        f"{MIN_PALABRAS_RESUMEN} palabras; {len(excluidas)} de enseñanza excluidas; {len(analisis)} para análisis"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
