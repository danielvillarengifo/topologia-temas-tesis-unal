"""Cuenta las palabras del texto corrido del cuerpo de cada tesis.

Lee:     datos/texto.nosync/<uuid>.txt
Escribe: datos/conteos_cuerpo.csv.gz  (uuid, palabra, veces), solo palabras frecuentes
         datos/palabras_por_tesis.csv (uuid, palabras y bloques de texto corrido)

Los textos completos no van al repositorio; estos conteos sí, y bastan para
rehacer el análisis de vocabulario.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from extraer_parrafos import bloques_de_texto

MIN_TESIS_POR_PALABRA = 10
MIN_PALABRAS = 2000


def main(carpeta_texto, carpeta_datos):
    conteos, totales = {}, []
    for archivo in sorted(carpeta_texto.glob("*.txt")):
        bloques = bloques_de_texto(archivo.read_text(encoding="utf-8", errors="replace"))
        palabras = re.findall(r"[a-záéíóúüñ]{2,}", " ".join(bloques).lower())
        if len(palabras) < MIN_PALABRAS:
            continue
        conteos[archivo.stem] = Counter(palabras)
        totales.append({"uuid": archivo.stem, "palabras": len(palabras), "bloques": len(bloques)})

    tesis_por_palabra = Counter(palabra for conteo in conteos.values() for palabra in conteo)
    frecuentes = {p for p, n in tesis_por_palabra.items() if n >= MIN_TESIS_POR_PALABRA}
    filas = [
        (uuid, palabra, veces)
        for uuid, conteo in conteos.items()
        for palabra, veces in conteo.items()
        if palabra in frecuentes
    ]
    pd.DataFrame(filas, columns=["uuid", "palabra", "veces"]).to_csv(carpeta_datos / "conteos_cuerpo.csv.gz", index=False)
    pd.DataFrame(totales).to_csv(carpeta_datos / "palabras_por_tesis.csv", index=False)
    print(f"{len(conteos)} tesis, {len(frecuentes)} palabras distintas")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
