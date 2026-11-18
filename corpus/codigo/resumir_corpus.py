"""Cifras agregadas del corpus, que es lo único que los artículos leen de él.

Lee de datos/:    metadatos_crudos.jsonl, tesis.csv, tesis_excluidas.csv,
                  texto_descargas.csv, parrafos_cuerpo.csv, palabras_por_tesis.csv
Escribe en datos/:
  resumen_corpus.csv      una cifra por fila (clave, valor)
  corpus_por_anio.csv     tesis por año y área
  corpus_por_periodo.csv  la tabla descriptiva de los artículos
"""
import sys
from pathlib import Path

import pandas as pd

ULTIMO_ANIO_PRE_IA = 2021


def periodo(anio):
    if anio <= ULTIMO_ANIO_PRE_IA:
        return "Hasta 2021"
    return "2022" if anio == 2022 else "2023 a 2026"


def main(datos):
    tesis = pd.read_csv(datos / "tesis.csv")
    descargas = pd.read_csv(datos / "texto_descargas.csv")
    parrafos = pd.read_csv(datos / "parrafos_cuerpo.csv")
    texto_corrido = pd.read_csv(datos / "palabras_por_tesis.csv")
    with open(datos / "metadatos_crudos.jsonl", encoding="utf-8") as crudo:
        registros = sum(1 for _ in crudo)

    # Algunas tesis se bajaron antes de quedar excluidas del análisis.
    descargas = descargas[descargas["uuid"].isin(tesis["uuid"])]
    tesis["periodo"] = tesis["anio"].map(periodo)
    tesis["con_parrafos"] = tesis["uuid"].isin(parrafos["uuid"])
    tesis["con_texto_corrido"] = tesis["uuid"].isin(texto_corrido["uuid"])

    cifras = {
        "registros_bajados": registros,
        "tesis_de_ensenanza_excluidas": len(pd.read_csv(datos / "tesis_excluidas.csv")),
        "tesis_de_analisis": len(tesis),
        "textos_pedidos": len(descargas),
        "textos_obtenidos": int((descargas["estado"] == "ok").sum()),
        "tesis_con_parrafos": int(tesis["con_parrafos"].sum()),
        "parrafos": int(parrafos["uuid"].isin(tesis["uuid"]).sum()),
        "tesis_con_texto_corrido": int(tesis["con_texto_corrido"].sum()),
        "mediana_palabras_texto_corrido": int(texto_corrido.loc[texto_corrido["uuid"].isin(tesis["uuid"]), "palabras"].median()),
        "mediana_palabras_resumen": int(tesis["n_palabras_resumen"].median()),
    }
    pd.Series(cifras, name="valor").rename_axis("clave").to_csv(datos / "resumen_corpus.csv")

    tesis.groupby(["anio", "area"], as_index=False).agg(
        tesis=("uuid", "size"), con_parrafos=("con_parrafos", "sum"), con_texto_corrido=("con_texto_corrido", "sum")
    ).to_csv(datos / "corpus_por_anio.csv", index=False)

    orden = ["Hasta 2021", "2022", "2023 a 2026"]
    tesis.groupby("periodo").agg(
        tesis=("uuid", "size"),
        estadistica=("area", lambda a: (a == "Estadística").sum()),
        matematicas=("area", lambda a: (a == "Matemáticas").sum()),
        doctorado=("nivel", lambda n: (n == "Doctorado").sum()),
        con_parrafos=("con_parrafos", "sum"),
        mediana_palabras_resumen=("n_palabras_resumen", "median"),
    ).loc[orden].round().astype(int).to_csv(datos / "corpus_por_periodo.csv")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
