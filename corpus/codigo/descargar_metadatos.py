"""Baja los metadatos de las tesis de maestría y doctorado en Estadística y
Matemáticas del repositorio de la UNAL (DSpace 7).

Salida: datos/metadatos_crudos.jsonl, un ítem por línea con todos sus campos.
"""
import json
import sys
import time
from pathlib import Path

import requests

API = "https://bffrepositorio.unal.edu.co/server/api"
TIPOS = '(dc.type:"Trabajo de grado - Maestría" OR dc.type:"Trabajo de grado - Doctorado")'
# El área no tiene un campo único: las tesis migradas de bdigital la traen en
# dc.relation.ispartof y las nuevas en dc.publisher.program o degreename.
CAMPOS_AREA = [
    "dc.publisher.program",
    "dc.description.degreename",
    "dc.relation.ispartof",
    "dc.publisher.department",
]
RAICES = ["Estadística", "Matemática", "Matemáticas"]
PAUSA_S = 0.5
TAMANO_PAGINA = 100


def paginas(sesion, consulta):
    pagina = 0
    while True:
        respuesta = sesion.get(
            f"{API}/discover/search/objects",
            params={"query": consulta, "dsoType": "ITEM", "size": TAMANO_PAGINA, "page": pagina},
            timeout=120,
        )
        respuesta.raise_for_status()
        resultado = respuesta.json()["_embedded"]["searchResult"]
        for objeto in resultado.get("_embedded", {}).get("objects", []):
            yield objeto["_embedded"]["indexableObject"]
        pagina += 1
        if pagina >= resultado["page"]["totalPages"]:
            return
        time.sleep(PAUSA_S)


def main(destino):
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "corpus-tesis-unal/1.0 (trabajo academico)"
    items = {}
    for campo in CAMPOS_AREA:
        for raiz in RAICES:
            antes = len(items)
            for item in paginas(sesion, f"{TIPOS} AND {campo}:*{raiz}*"):
                items[item["uuid"]] = {
                    "uuid": item["uuid"],
                    "handle": item["handle"],
                    "metadata": item["metadata"],
                }
            print(f"{campo}:*{raiz}*  nuevos: {len(items) - antes}  total: {len(items)}")
    assert items, "la búsqueda no devolvió ninguna tesis"
    with open(destino, "w", encoding="utf-8") as salida:
        for item in items.values():
            salida.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(items)} tesis en {destino}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
