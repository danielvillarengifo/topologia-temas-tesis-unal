"""Cambios de vocabulario en el cuerpo y en el resumen de las tesis.

Para cada palabra compara su tasa por 10 000 palabras, tesis por tesis, entre
dos ventanas de años. La comparación entre dos ventanas anteriores a la IA da
cuántas palabras cambian por deriva ordinaria. Las dos comparaciones prueban
la misma familia de palabras.

El conteo se repite con controles, porque un cambio de vocabulario también
puede venir de tesis más largas, de otra mezcla de áreas o de temas nuevos:
  todas las tesis
  longitud fija          cada tesis reducida al azar al mismo número de palabras
  Estadística            solo esa área
  Matemáticas            solo esa área
  sin aprendizaje automático   sin las tesis de ese tema

Lee:  corpus/datos/tesis.csv, conteos_cuerpo.csv.gz, palabras_por_tesis.csv
Escribe en resultados/:
  vocabulario_cambios.csv   por comparación y palabra, con todas las tesis: tasas, razón, estadístico y valor q
  vocabulario_resumen.csv   por control y comparación: tesis, palabras probadas y cuántas suben o bajan
  marcadores_por_anio.csv   tasa anual de las raíces marcadoras, por serie y grupo de tesis, con intervalo
  marcadores_raices.csv     cambio de cada raíz marcadora entre ventanas
  logodds_resumenes.csv     palabras más propias de cada ventana en los resúmenes
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, norm
from sklearn.feature_extraction.text import CountVectorizer

from corpus import PALABRAS_VACIAS, SEMILLA, cargar_tesis
from variables import RAICES_CON_USO_TECNICO, RAICES_MARCADORAS

# El texto completo existe desde 2016, así que las ventanas del cuerpo son de tres, tres y cuatro años.
VENTANAS_CUERPO = {"2016-2018": (2016, 2018), "2019-2021": (2019, 2021), "2023-2026": (2023, 2026)}
COMPARACIONES_CUERPO = {
    "sin IA de por medio": ("2016-2018", "2019-2021"),
    "con IA de por medio": ("2019-2021", "2023-2026"),
}
VENTANAS_RESUMEN = ("2018-2021", "2023-2026")
MIN_PRESENCIA = 0.20
NIVEL_Q = 0.05
REPLICAS = 2000
POR_DIEZ_MIL = 10_000
PALABRAS_FIJAS = 4000
PALABRAS_EN_NUBE = 40
# Evita razones infinitas cuando una palabra no aparece en una ventana.
SUAVIZADO_DE_TASA = 0.5
# Peso total del prior de Dirichlet, en palabras: contrae las palabras raras hacia cero.
PESO_DEL_PRIOR = 500


def conteos_por_tesis(conteos, totales):
    """Matriz tesis x palabra de conteos. La última columna reúne las palabras
    raras que el archivo de conteos no trae una por una."""
    ancha = conteos.pivot(index="uuid", columns="palabra", values="veces").reindex(totales.index).fillna(0).astype(int)
    ancha["(otras)"] = totales["palabras"] - ancha.sum(axis=1)
    return ancha


def reducir_a_longitud_fija(conteos, azar):
    """Deja cada tesis con el mismo número de palabras, sacándolas al azar sin
    reemplazo. Las tesis más cortas que eso no entran."""
    largas = conteos[conteos.sum(axis=1) >= PALABRAS_FIJAS]
    reducidas = np.vstack([azar.multivariate_hypergeometric(fila, PALABRAS_FIJAS) for fila in largas.to_numpy()])
    return pd.DataFrame(reducidas, index=largas.index, columns=largas.columns)


def tasas(conteos):
    return conteos.drop(columns="(otras)").div(conteos.sum(axis=1), axis=0) * POR_DIEZ_MIL


def comparar(tasas_a, tasas_b):
    # Una palabra ausente en las dos ventanas de un control no tiene prueba: vale como sin cambio.
    prueba = mannwhitneyu(tasas_b, tasas_a, axis=0, method="asymptotic")
    valor_p = np.nan_to_num(prueba.pvalue, nan=1.0)
    # El sentido es el de la prueba, que compara rangos y no medias.
    sube = prueba.statistic > len(tasas_a) * len(tasas_b) / 2
    return pd.DataFrame({
        "palabra": tasas_a.columns,
        "tasa_a": tasas_a.mean().to_numpy(),
        "tasa_b": tasas_b.mean().to_numpy(),
        "log2_razon": np.log2((tasas_b.mean() + SUAVIZADO_DE_TASA) / (tasas_a.mean() + SUAVIZADO_DE_TASA)).to_numpy(),
        # El z sale del valor p, que ya corrige los empates: ordena igual que él.
        "z": np.where(sube, 1, -1) * norm.isf(valor_p / 2),
        "valor_p": valor_p,
        "valor_q": false_discovery_control(valor_p),
    })


def tasa_anual_con_intervalo(tasa, anio, azar):
    filas = []
    for a in sorted(set(anio)):
        del_anio = tasa[anio == a].to_numpy()
        medias = del_anio[azar.integers(len(del_anio), size=(REPLICAS, len(del_anio)))].mean(axis=1)
        filas.append({"anio": a, "n_tesis": len(del_anio), "tasa": del_anio.mean(),
                      "inf": np.quantile(medias, 0.025), "sup": np.quantile(medias, 0.975)})
    return pd.DataFrame(filas)


def logodds_ponderado(documentos_a, documentos_b):
    """Razón de momios logarítmica con prior de Dirichlet informativo (Monroe,
    Colaresi y Quinn, 2008): z positivo es palabra más propia de la ventana b."""
    contador = CountVectorizer(token_pattern=r"(?u)\b[a-záéíóúüñ]{3,}\b", stop_words=PALABRAS_VACIAS, min_df=5)
    conteos = contador.fit_transform(pd.concat([documentos_a, documentos_b]))
    y_a = np.asarray(conteos[:len(documentos_a)].sum(axis=0)).ravel()
    y_b = np.asarray(conteos[len(documentos_a):].sum(axis=0)).ravel()
    prior = (y_a + y_b) / (y_a + y_b).sum() * PESO_DEL_PRIOR
    logit_a = np.log((y_a + prior) / (y_a.sum() + prior.sum() - y_a - prior))
    logit_b = np.log((y_b + prior) / (y_b.sum() + prior.sum() - y_b - prior))
    z = (logit_b - logit_a) / np.sqrt(1 / (y_a + prior) + 1 / (y_b + prior))
    return pd.DataFrame({"palabra": contador.get_feature_names_out(), "z": z, "veces_a": y_a, "veces_b": y_b})


def empieza_por(palabras, raices):
    return np.array([p.startswith(tuple(raices)) for p in palabras])


def main(carpeta_corpus, carpeta):
    tesis = cargar_tesis(carpeta_corpus / "tesis.csv").set_index("uuid")
    totales = pd.read_csv(carpeta_corpus / "palabras_por_tesis.csv").set_index("uuid")
    totales = totales[totales.index.isin(tesis.index)]
    conteos = conteos_por_tesis(pd.read_csv(carpeta_corpus / "conteos_cuerpo.csv.gz"), totales)
    datos = tesis.loc[conteos.index, ["anio", "area", "es_de_aprendizaje_automatico"]]
    datos["ventana"] = ""
    for nombre, (desde, hasta) in VENTANAS_CUERPO.items():
        datos.loc[datos["anio"].between(desde, hasta), "ventana"] = nombre
    azar = np.random.default_rng(SEMILLA)

    todas = tasas(conteos)
    # Una sola familia de palabras para las dos comparaciones: las presentes en
    # al menos una de cada cinco tesis de las tres ventanas juntas.
    familia = todas.columns[(todas[datos["ventana"] != ""] > 0).mean() >= MIN_PRESENCIA]
    controles = {
        "todas las tesis": todas,
        "longitud fija": tasas(reducir_a_longitud_fija(conteos, azar)),
        "Estadística": todas[datos["area"] == "Estadística"],
        "Matemáticas": todas[datos["area"] == "Matemáticas"],
        "sin aprendizaje automático": todas[~datos["es_de_aprendizaje_automatico"]],
    }

    cambios, resumen = [], []
    for control, tabla in controles.items():
        for comparacion, (ventana_a, ventana_b) in COMPARACIONES_CUERPO.items():
            en_a = tabla[datos.loc[tabla.index, "ventana"] == ventana_a][familia]
            en_b = tabla[datos.loc[tabla.index, "ventana"] == ventana_b][familia]
            resultado = comparar(en_a, en_b)
            etiquetas = {"comparacion": comparacion, "ventana_a": ventana_a, "ventana_b": ventana_b}
            if control == "todas las tesis":
                cambios.append(resultado.assign(
                    **etiquetas,
                    es_marcadora=empieza_por(resultado["palabra"], RAICES_MARCADORAS),
                    con_uso_tecnico=empieza_por(resultado["palabra"], RAICES_CON_USO_TECNICO),
                ))
            resumen.append({
                "control": control, **etiquetas, "n_tesis_a": len(en_a), "n_tesis_b": len(en_b),
                "palabras_probadas": len(familia),
                "suben": int(((resultado["valor_q"] < NIVEL_Q) & (resultado["z"] > 0)).sum()),
                "bajan": int(((resultado["valor_q"] < NIVEL_Q) & (resultado["z"] < 0)).sum()),
                "p_menor_que_005": float((resultado["valor_p"] < 0.05).mean()),
            })
    pd.concat(cambios).to_csv(carpeta / "vocabulario_cambios.csv", index=False)
    pd.DataFrame(resumen).to_csv(carpeta / "vocabulario_resumen.csv", index=False)

    palabras = todas.columns
    series = {
        "todas las raíces": empieza_por(palabras, RAICES_MARCADORAS),
        "sin uso técnico": empieza_por(palabras, [r for r in RAICES_MARCADORAS if r not in RAICES_CON_USO_TECNICO]),
    }
    anuales = []
    for serie, columnas in series.items():
        tasa = todas.loc[:, columnas].sum(axis=1)
        grupos = {
            "todas": np.ones(len(datos), dtype=bool),
            "Estadística": (datos["area"] == "Estadística").to_numpy(),
            "Matemáticas": (datos["area"] == "Matemáticas").to_numpy(),
            "sin aprendizaje automático": ~datos["es_de_aprendizaje_automatico"].to_numpy(),
        }
        for grupo, filas in grupos.items():
            anuales.append(tasa_anual_con_intervalo(tasa[filas], datos.loc[filas, "anio"], azar).assign(serie=serie, area=grupo))
    pd.concat(anuales).to_csv(carpeta / "marcadores_por_anio.csv", index=False)

    por_raiz = pd.DataFrame({r: todas.loc[:, empieza_por(palabras, [r])].sum(axis=1) for r in RAICES_MARCADORAS})
    raices = []
    for comparacion, (ventana_a, ventana_b) in COMPARACIONES_CUERPO.items():
        resultado = comparar(por_raiz[datos["ventana"] == ventana_a], por_raiz[datos["ventana"] == ventana_b])
        raices.append(resultado.rename(columns={"palabra": "raiz"}).assign(
            comparacion=comparacion, con_uso_tecnico=lambda t: t["raiz"].isin(RAICES_CON_USO_TECNICO)
        ))
    pd.concat(raices).to_csv(carpeta / "marcadores_raices.csv", index=False)

    tesis = tesis.reset_index()
    ventana_a, ventana_b = VENTANAS_RESUMEN
    propias = logodds_ponderado(tesis.loc[tesis["ventana"] == ventana_a, "documento"], tesis.loc[tesis["ventana"] == ventana_b, "documento"])
    extremos = pd.concat([propias.nsmallest(PALABRAS_EN_NUBE, "z"), propias.nlargest(PALABRAS_EN_NUBE, "z")])
    extremos.assign(ventana=np.where(extremos["z"] > 0, ventana_b, ventana_a)).to_csv(carpeta / "logodds_resumenes.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
