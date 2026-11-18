"""Baja el PDF de cada tesis que no está en inglés desde un año dado, extrae el texto con
pdftotext y borra el PDF.

Salida: datos/texto.nosync/<uuid>.txt y datos/texto_descargas.csv con el estado
de cada tesis. Se puede interrumpir y volver a correr: salta lo ya extraído.
Con «recientes» como cuarto argumento empieza por las tesis más nuevas, lo que
permite correr dos procesos a la vez, uno desde cada extremo.
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://bffrepositorio.unal.edu.co/server/api"
PAUSA_S = 1.0


def enlace_pdf(sesion, uuid):
    respuesta = sesion.get(f"{API}/core/items/{uuid}/bundles", params={"embed": "bitstreams"}, timeout=120)
    respuesta.raise_for_status()
    for bundle in respuesta.json()["_embedded"]["bundles"]:
        if bundle["name"] != "ORIGINAL":
            continue
        archivos = bundle["_embedded"]["bitstreams"]["_embedded"]["bitstreams"]
        pdfs = [a for a in archivos if a["name"].lower().endswith(".pdf")]
        if pdfs:
            return max(pdfs, key=lambda a: a["sizeBytes"])["_links"]["content"]["href"]
    return None


def extraer(sesion, uuid, carpeta_pdf, carpeta_texto):
    enlace = enlace_pdf(sesion, uuid)
    if enlace is None:
        return "sin_pdf", 0
    pdf = carpeta_pdf / f"{uuid}.pdf"
    with sesion.get(enlace, stream=True, timeout=300) as descarga:
        if descarga.status_code in (401, 403):
            return "restringido", 0
        descarga.raise_for_status()
        with open(pdf, "wb") as destino:
            for bloque in descarga.iter_content(1 << 20):
                destino.write(bloque)
    texto = carpeta_texto / f"{uuid}.txt"
    conversion = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(texto)], capture_output=True)
    pdf.unlink()
    if conversion.returncode != 0:
        texto.unlink(missing_ok=True)
        return "pdf_ilegible", 0
    return "ok", texto.stat().st_size


def main(tabla, carpeta_datos, anio_inicial, recientes_primero):
    carpeta_pdf = carpeta_datos / "crudo.nosync"
    carpeta_texto = carpeta_datos / "texto.nosync"
    carpeta_pdf.mkdir(exist_ok=True)
    carpeta_texto.mkdir(exist_ok=True)
    bitacora = carpeta_datos / "texto_descargas.csv"
    hechas = set(pd.read_csv(bitacora)["uuid"]) if bitacora.exists() else set()

    tesis = pd.read_csv(tabla)
    pendientes = tesis[(tesis["anio"] >= anio_inicial) & (tesis["idioma"] != "eng") & ~tesis["uuid"].isin(hechas)]
    if recientes_primero:
        pendientes = pendientes.iloc[::-1]
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "corpus-tesis-unal/1.0 (trabajo academico)"

    with open(bitacora, "a", newline="", encoding="utf-8") as archivo:
        registro = csv.writer(archivo)
        if not hechas:
            registro.writerow(["uuid", "estado", "bytes_texto"])
        for numero, uuid in enumerate(pendientes["uuid"], start=1):
            if (carpeta_texto / f"{uuid}.txt").exists():
                continue
            try:
                estado, tamano = extraer(sesion, uuid, carpeta_pdf, carpeta_texto)
            except requests.RequestException as error:
                # Un fallo de red no debe tumbar una corrida de una hora; queda
                # en la bitácora y se reintenta borrando esa fila.
                estado, tamano = f"error_red: {type(error).__name__}", 0
            registro.writerow([uuid, estado, tamano])
            archivo.flush()
            if numero % 25 == 0:
                print(f"{numero}/{len(pendientes)}", flush=True)
            time.sleep(PAUSA_S)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:] == ["recientes"])
