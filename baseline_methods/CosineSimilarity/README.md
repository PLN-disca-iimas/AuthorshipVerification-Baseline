## Datos

***

Los corpus de entrenamiento y prueba se enceuntran en la carpeta 'corpus' en la raíz del proyecto.

## Prerrequisitos

***
  - Python 3.6+ (se recomienda Anaconda Python distribution)
  - scikit-learn, numpy, scipy
  - non-essential: tqdm, seaborn/matplotlib
  - verif_evaluator.py
  - split.py

## Ejecutar

***
> __Se asume que HOME_PATH es la ruta en donde se encuentra el proyecto clonado.__ 
  
Para ejecutar el script, usando los datos del pan14:
  ```sh
	>> python main.py \
	          -input_pairs="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan14/train.jsonl" \
	          -input_truth="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan14/train_truth.jsonl" \
	          -test_pairs="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan14/test.jsonl" \
	          -test_truth="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan14/test_truth.jsonl" \
	          -output="pan14_pred.jsonl"
  ```

Para ejecutar el script, usando los datos del pan15:
  ```sh
	>> python main.py \
	          -input_pairs="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan15/train.jsonl" \
	          -input_truth="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan15/train_truth.jsonl" \
	          -test_pairs="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan15/test.jsonl" \
	          -test_truth="/HOME_PATH/AuthorshipVerification-Baseline/corpus/pan15/test_truth.jsonl" \
	          -output="pan15_pred.jsonl"
  ```


Salida:
  ```sh
	optimal p1/p2: 0.73 0.79
	optimal score: {'auc': 0.889, 'c@1': 0.832, 'f_05_u': 0.796, 'F1': 0.9, 'brier': 0.803, 'overall': 0.844}
	-> determining optimal threshold
	Dev results -> F1=0.7999999999999999 at th=0.5002502502502503
	comenzando con los plot
	se creo pdf
	-> calculating test similarities
	200it [00:05, 36.91it/s]
  ```

| Args   		  		| Description                                    |
|-----------------|------------------------------------------------|
| `-test_pairs`   | test.jsonl archivo con ruta absoluta           |
| `-input_truth`  | input_truth.jsonl archivo con ruta absoluta    |
| `-input_pairs`  | train.jsonl archivo con ruta absoluta          |
| `-input_truth`  | train_truth.jsonl archivo con ruta absoluta    |
| `-output`   	  | nombre del archivo de salida	               	 |