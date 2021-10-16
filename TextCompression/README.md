# Compression method calculating cross-entropy


El directorio src contiene dos archivos: text_compression.py y evaluator_PAN21.py. El primero es el baseline per se, y el segundo es el evaluador con las métricas propuestas para el PAN2021.

Para ejecutar el baseline desde la línea de comandos necesitamos las siguientes banderas:
- - i ruta del archivo de pruebas
- - o directorio donde se están los resultados

```
python text_compression.py -i EVALUATION-FILE -o OUTPUT-DIRECTORY [-m MODEL-FILE]
```

La bandera -m indica a ruta del modelo (de regresión) entrenado a usar para la clasificación de los textos a partir de sus entropías cruzadas.


Nota: el argumento con bandera -i debe terminal con .json, mientras que el argumento con bandera -o es un directorio.


