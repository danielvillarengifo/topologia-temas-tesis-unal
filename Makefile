# `make` compila el artículo y su versión en inglés con los resultados incluidos.
# `make corpus` baja las tesis de la UNAL (cerca de una hora) y `make todo` rehace el análisis.
PYTHON ?= python3
export LC_ALL = en_US.UTF-8
# Typst graba la fecha de compilación; con esta fija, el PDF sale igual byte a byte.
export SOURCE_DATE_EPOCH = 1794960000
PY = PYTHONPATH=corpus/codigo:codigo TOKENIZERS_PARALLELISM=false $(PYTHON) -W ignore
C = corpus/datos

articulo: topologia.pdf topologia_en.pdf

topologia.pdf: topologia.qmd codigo/cifras.R referencias.bib
	quarto render topologia.qmd --to typst

topologia_en.pdf: topologia_en.qmd codigo/cifras.R referencias.bib
	quarto render topologia_en.qmd --to typst

corpus:
	$(PYTHON) corpus/codigo/descargar_metadatos.py $(C)/metadatos_crudos.jsonl
	$(PYTHON) corpus/codigo/construir_tabla.py $(C)/metadatos_crudos.jsonl $(C)/tesis.csv $(C)/tesis_excluidas.csv
	$(PYTHON) corpus/codigo/extraer_texto.py $(C)/tesis.csv $(C) 2016
	$(PYTHON) corpus/codigo/extraer_parrafos.py $(C)/texto.nosync $(C)/parrafos_cuerpo.csv
	$(PYTHON) corpus/codigo/contar_palabras.py $(C)/texto.nosync $(C)
	$(PYTHON) corpus/codigo/resumir_corpus.py $(C)

todo:
	$(PY) codigo/mapa_tematico.py $(C)/tesis.csv resultados
	$(PY) codigo/clasificar_periodo.py $(C)/tesis.csv resultados/clasificacion_periodo.csv
	$(PY) codigo/homologia.py $(C)/tesis.csv resultados
	$(PY) codigo/vocabulario.py $(C) resultados
	Rscript codigo/figuras.R resultados figuras es
	Rscript codigo/figuras.R resultados figuras/en en
	quarto render topologia.qmd --to typst
	quarto render topologia_en.qmd --to typst

.PHONY: articulo corpus todo
