# Baseline 

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del [PAN2021](https://pan.webis.de/clef21/pan21-web/author-identification.html).

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.


## Prerequisites

***
  - Using venv

  > Make sure `pipenv` is installed. (If not, simply run: `pip install pipenv`.)
  ```sh
  # Create the virtual environment
  python3.9 -m venv env
  # Activate the virtual environment
  source env/bin/activate
  # Install the dependencies
  pipenv install -r requirements.txt
  ```
## corpus

***
Los archivos vienen separados  con forme a su respectivo pan, digamos pan14 y pan15.
Lo cual dentro de cada carpeta se encuentran siguientes archivos .jsonl.
- test.jsonl
- test_truth.jsonl
- trian.jsonl
- train_truth.jsonl

Tanto el conjunto de datos pequeño como el grande vienen con dos archivos JSON delimitados por saltos de línea cada uno (*.jsonl). El primer archivo contiene partes de textos (cada par tiene una identificación única) y sus etiquetas de fondo:

```
test.jsonl

{"id": "EN101", "pair": ["Only when the Nan-yang Maru sailed from Yuen-San did her terrible sense of foreboding begin to subside. For four years, waking or sleeping, the awful subconsciousness of supreme evil had never left her. But now, as the Korean shore, receding into darkness, grew dimmer and dimmer, fear subsided and grew vague as the half-forgotten memory of horror in a dream. She stood near the steamer's stern apart from other passengers, a slender, lonely figure in her silver-fox furs, her ulster and smart little hat, watching the lights of Yuen-San grow paler and smaller along the horizon until they looked like a level row of stars. Under her haunted eyes Asia was slowly dissolving to a streak of vapour in the misty lustre of the moon. Suddenly the ancient continent disappeared, washed out by a wave against the sky; and with it vanished the last shreds of that accursed nightmare which had possessed her for four endless years. But whether during those unreal years her soul had only been held in bondage, or whether, as she had been taught, it had been irrevocably destroyed, she still remained uncertain, knowing nothing about the death of souls or how it was accomplished. As she stood there, her sad eyes fixed on the misty East, a passenger passing--an Englishwoman--paused to say something kind to the young American; and added, \"if there is anything my husband and I can do it would give us much pleasure.\" The girl had turned her head as though not comprehending. The other woman hesitated. \"This is Doctor Norne's daughter, is it not?\" she inquired in a pleasant voice. \"Yes, I am Tressa Norne.... I ask your pardon.... Thank 
```
y por ejemplo tenemos para *_truth.jsonl, lo cual se muestra de la siguiente manera:

```
{"id": "EN101", "value": false}
{"id": "EN102", "value": false}
{"id": "EN103", "value": true}
{"id": "EN104", "value": false}
{"id": "EN105", "value": false}
{"id": "EN106", "value": false}
{"id": "EN107", "value": false}
{"id": "EN108", "value": true}
{"id": "EN109", "value": false}
{"id": "EN110", "value": true}
```



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

