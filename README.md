# Baseline 

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del [PAN2021](https://pan.webis.de/clef21/pan21-web/author-identification.html).

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.

En esta carpeta se encuentran los codigos necesarios para poder correr los metodos que ocupamos siendo  compression method y Support Vector Classification  con parámetros predeterminados para realizar una comparación con otros modelos para la tarea de verificación de autoría del PAN2021.

## corpus

***
Los archivos vienen separados 

Tanto el conjunto de datos pequeño como el grande vienen con dos archivos JSON delimitados por saltos de línea cada uno (*.jsonl). El primer archivo contiene pares de textos (cada par tiene una identificación única) y sus etiquetas de fandom:



## papers
***
En esta carpeta se encuentra un archivo pdf, con el cual fue un apoyo para la implementacion de dicho metodo de compresion.
 - Language Modeling for information Retrieval


## utils 
***
La carpeta utils contiene codigos .py que nos ayudan a poder correr los metodos principales y obtener los datos que necesitamos, siendo:

 - split.py
 - verif_evaluator.py


## Prerequisites

***
  - Usa pipenv

  > Asegúrate de que `pipenv` esté instalado. (Si no, simplemente ejecute: `pip install pipenv`).
  ```sh
  # Activar el entorno virtual
  pipenv shell
  # Instalar las dependencias
  pipenv install -r requirements.txt
  ```

