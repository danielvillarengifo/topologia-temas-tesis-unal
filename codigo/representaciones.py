"""Representaciones de un texto que dependen de un modelo de lenguaje preentrenado.

  incrustaciones:  vector de 768 dimensiones de un codificador congelado:
                   multilingual-e5-base (multilingüe) o BETO (solo español).
  verosimilitud:   tres estadísticos de qué tan previsible es el texto para un
                   GPT-2 entrenado en español: log-probabilidad media, log-rango
                   medio y entropía media de la predicción.

Calcularlas tarda minutos, así que se guardan en datos/cache.nosync/ de la
carpeta desde la que se corre, con la huella del texto como llave. Cada
artículo tiene así su propio caché, que no va al repositorio. La llave no
incluye el modelo ni MAX_FICHAS: si se cambian, hay que borrar el caché.
Los vectores dependen del dispositivo en el séptimo decimal.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Nombre del modelo y prefijo con el que se entrenó (E5 lo exige; BETO no usa).
CODIFICADORES = {
    "e5": ("intfloat/multilingual-e5-base", "query: "),
    "beto": ("dccuchile/bert-base-spanish-wwm-cased", ""),
}
MODELO_GENERATIVO = "DeepESP/gpt2-spanish"
MAX_FICHAS = 384
LOTE = 8
DISPOSITIVO = "mps" if torch.backends.mps.is_available() else "cpu"
CACHE = Path("datos/cache.nosync")


def huella(texto):
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def con_cache(nombre, calcular, textos):
    """Devuelve una fila por texto, calculando solo los que no están en el caché."""
    CACHE.mkdir(parents=True, exist_ok=True)
    archivo = CACHE / f"{nombre}.parquet"
    guardado = pd.read_parquet(archivo) if archivo.exists() else pd.DataFrame()
    llaves = textos.map(huella)
    nuevos = textos[~llaves.isin(guardado.index)].drop_duplicates()
    if len(nuevos):
        calculado = calcular(list(nuevos))
        calculado.index = nuevos.map(huella)
        guardado = pd.concat([guardado, calculado])
        guardado.to_parquet(archivo)
    return guardado.loc[llaves].set_index(textos.index)


@torch.no_grad()
def calcular_incrustaciones(textos, codificador):
    nombre, prefijo = CODIFICADORES[codificador]
    fichador = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModel.from_pretrained(nombre).to(DISPOSITIVO).eval()
    vectores = []
    for inicio in range(0, len(textos), LOTE):
        lote = [prefijo + t for t in textos[inicio:inicio + LOTE]]
        fichas = fichador(lote, padding=True, truncation=True, max_length=MAX_FICHAS, return_tensors="pt").to(DISPOSITIVO)
        salida = modelo(**fichas).last_hidden_state
        mascara = fichas["attention_mask"].unsqueeze(-1)
        media = (salida * mascara).sum(dim=1) / mascara.sum(dim=1)
        vectores.append(torch.nn.functional.normalize(media, dim=1).cpu().numpy())
    vectores = np.vstack(vectores)
    return pd.DataFrame(vectores, columns=[f"{codificador}_{i:03d}" for i in range(vectores.shape[1])])


@torch.no_grad()
def calcular_verosimilitud(textos):
    fichador = AutoTokenizer.from_pretrained(MODELO_GENERATIVO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_GENERATIVO).to(DISPOSITIVO).eval()
    filas = []
    for texto in textos:
        fichas = fichador(texto, truncation=True, max_length=MAX_FICHAS, return_tensors="pt").input_ids.to(DISPOSITIVO)
        log_prob = torch.log_softmax(modelo(fichas).logits[0, :-1], dim=-1)
        siguientes = fichas[0, 1:]
        de_la_ficha = log_prob.gather(1, siguientes.unsqueeze(1)).squeeze(1)
        rango = (log_prob > de_la_ficha.unsqueeze(1)).sum(dim=1) + 1
        entropia = -(log_prob.exp() * log_prob).sum(dim=1)
        filas.append({
            "z_log_prob": de_la_ficha.mean().item(),
            "z_log_rango": rango.float().log().mean().item(),
            "z_entropia": entropia.mean().item(),
        })
    return pd.DataFrame(filas)


def incrustaciones(textos, codificador="e5"):
    return con_cache(f"incrustaciones_{codificador}", lambda nuevos: calcular_incrustaciones(nuevos, codificador), textos)


def verosimilitud(textos):
    return con_cache("verosimilitud", calcular_verosimilitud, textos)


if __name__ == "__main__":
    prueba = pd.Series([
        "En este trabajo se estima un modelo de regresión con datos de la región.",
        "Verde ideas incoloras duermen furiosamente sobre matrices de la tarde.",
    ])
    vectores = calcular_incrustaciones(list(prueba), "e5").to_numpy()
    previsible, absurdo = calcular_verosimilitud(list(prueba))["z_log_prob"]
    assert vectores.shape == (2, 768) and np.allclose(np.linalg.norm(vectores, axis=1), 1, atol=1e-4)
    assert previsible > absurdo, "una frase corriente debe ser más previsible que una absurda"
    print(f"autochequeo correcto en {DISPOSITIVO}")
