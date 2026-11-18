"""Limpieza del texto de las tesis: repara lo que sale mal de un PDF hecho con
LaTeX (tildes separadas de su letra, ligaduras) y separa el español del inglés
en los resúmenes que traen los dos idiomas en un mismo campo."""
import re
import unicodedata

ACENTO_ANTES = {"´": "́", "˜": "̃", "¨": "̈"}
PALABRAS_ES = set("de la el en los las que se del un una para con por es son como este esta al su más entre".split())
PALABRAS_EN = set("the of and in to is for this that with are we on an as by be from which was were these has have".split())
# El repositorio añade esta nota al final de casi todos los resúmenes desde 2021
# y de pocos antes: si se deja, delata el año y no el estilo.
NOTA_DEL_REPOSITORIO = re.compile(
    r"\(\s*(?:textos?\s+)?tomados?\s+de\s+la\s+fu\w+\s*\)\.?|textos?\s+tomados?\s+de\s+la\s+fu\w+\.?", re.IGNORECASE
)


def reparar(texto):
    texto = texto.replace("ı", "i")
    texto = re.sub(r"([´˜¨])\s?([A-Za-z])", lambda m: m.group(2) + ACENTO_ANTES[m.group(1)], texto)
    return unicodedata.normalize("NFKC", texto)


def solo_espanol(texto):
    """Quita las oraciones en inglés. Entre 2009 y 2012 el repositorio guardó
    resumen y abstract juntos en muchas tesis."""
    conservadas = []
    for oracion in re.split(r"(?<=[.!?])\s+", texto):
        palabras = re.findall(r"[a-záéíóúñ]+", oracion.lower())
        if sum(p in PALABRAS_EN for p in palabras) <= sum(p in PALABRAS_ES for p in palabras):
            conservadas.append(oracion)
    return " ".join(conservadas)


def limpiar_resumen(texto):
    """Deja un resumen, humano o de IA, solo con su prosa en español."""
    texto = NOTA_DEL_REPOSITORIO.sub(" ", reparar(texto))
    texto = re.sub(r"\\[a-zA-Z]+|[$^_{}]", " ", texto)  # órdenes y signos de LaTeX
    texto = re.sub(r"\s+", " ", solo_espanol(texto)).strip()
    return texto if not texto or texto[-1] in ".!?" else texto + "."


if __name__ == "__main__":
    assert limpiar_resumen("Se estima el riesgo (Texto tomado de la fuente).") == "Se estima el riesgo."
    assert limpiar_resumen("Se estima el riesgo. (Texto tomado de la fuete)") == "Se estima el riesgo."
    assert limpiar_resumen("Los datos fueron tomados de la fuente oficial.") == "Los datos fueron tomados de la fuente oficial."
    assert limpiar_resumen("Sea $\\mathbb{R}^n$ el espacio.") == "Sea R n el espacio."
    assert solo_espanol("Se estima la función de riesgo. In this paper we study the problem.") == "Se estima la función de riesgo."
    assert reparar("Metodologı́a") == "Metodología"
    assert reparar("estad´ıstica y a˜no") == "estadística y año"
    assert reparar("clasiﬁcación") == "clasificación"
    print("autochequeo correcto")
