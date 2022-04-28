# Verificación de autoría PAN

La verificación de autoría es la tarea de decidir si dos textos han sido escritos por el mismo autor a partir de la comparación de los estilos de escritura de los textos.
La identificación de autoría en el [PAN](https://pan.webis.de/clef21/pan21-web/author-identification.html#task) espera contribuir en esta área y ha organizado tareas de identificación de autoría en varios aspectos.

# Baseline 

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del PAN2021.

- [Support Vector](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/baseline_methods/SupportVectorClassification)

Este modelo se basa en conjunto de ejemplos de entrenamiento (de muestras) podemos etiquetar las clases y entrenar una [SVM](https://en.wikipedia.org/wiki/Support-vector_machine) para construir un modelo que prediga la clase de una nueva muestra. Intuitivamente, una SVM es un modelo que representa a los puntos de muestra en el espacio, separando las clases a 2 espacios lo más amplios posibles mediante un hiperplano de separación definido como el vector entre los 2 puntos, de las 2 clases, más cercanos al que se llama vector soporte.

- [Cosine Similarity](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/baseline_methods/CosineSimilarity)

En esta carpeta se encuentra una solución rápida a la tarea PAN2020 sobre verificación de autoría. Todos los documentos se representan usando un modelo de Bag of character ngrams, eso es TFIDF ponderado. La [semejanza del coseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre cada par de documentos en el conjunto de datos de calibración es calculado. Finalmente, las similitudes resultantes son optimizadas y proyectadas a través de un simple reescalado, para que puedan funcionar como pseudo-probabilidades, que indican la probabilidad de que un par de documentos es un par del mismo autor.

- [Compresión de Textos](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/baseline_methods/TextCompression)

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.

# [Resultados](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/resultados)
***

| Corpus | Metodo                      | Train size | Test size | F1    | AUC   | Brier | c@1   | f_05_u | overall |
|--------|-----------------------------|------------|-----------|-------|-------|-------|-------|--------|--------|
| PAN14  |CosineSimilarity             | 100        | 200       | 0.669 | 0.684 | 0.748 | 0.504 | 0.562  | 0.633  |
| PAN14  |SupportVectorClassification  | 100        | 200       | 0.058 | 0.51  | 0.51  | 0.51  | 0.129  | 0.343  | 
| PAN14  |TextCompression              | 100        | 200       | 0.765 | 0.731 | 0.777 | 0.583 | 0.602  | 0.692  |
| PAN15  |CosineSimilarity             | 100        | 500       | 0.702 | 0.75  | 0.766 | 0.547 | 0.585  | 0.67   |
| PAN15  |SupportVectorClassification  | 100        | 500       | 0.653 | 0.588 | 0.588 | 0.588 | 0.597  | 0.603  |
| PAN15  |TextCompression              | 100        | 500       | 0.706 | 0.741 | 0.75  | 0.647 | 0.599  | 0.689  |
| PAN22  |CosineSimilarity             | 15732      | 1070      | 0.665 | 0.442 | 0.691 | 0.498 | 0.554  | 0.57   |
| PAN22  |SupportVectorClassification  | 15732      | 1070      | 0.696 | 0.575 | 0.575 | 0.575 | 0.594  | 0.603  |
| PAN22  |TextCompression              | 15732      | 1070      | 0.667 | 0.474 | 0.564 | 0.5   | 0.556  | 0.552  |

## Prerequisites
***
  - Using venv
  ```sh
  # Create the virtual environment
  python3.9 -m venv env
  # Activate the virtual environment
  source env/bin/activate
  # Install the dependencies
  pipenv install -r requirements.txt
  ```
## [corpus](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/corpus)

***
Los archivos vienen separados  con forme a su respectivo pan, digamos [Pan14](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/corpus) y [Pan15](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/corpus).
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
Corpus overview: [PAN14](https://pan.webis.de/downloads/publications/papers/stamatatos_2014.pdf), [PAN15](https://pan.webis.de/downloads/publications/papers/stamatatos_2015b.pdf)



## [papers](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/papers)
***
En esta carpeta se encuentra un archivo pdf, con el cual fue un apoyo para la implementacion de dicho metodo de compresion.
 - Language Modeling for information Retrieval


## [utils](https://github.com/PLN-disca-iimas/AuthorshipVerification-Baseline/tree/main/utils)
***
La carpeta utils contiene codigos .py que nos ayudan a poder correr los metodos principales y obtener los datos que necesitamos, siendo:

 - split.py
 - verif_evaluator.py



