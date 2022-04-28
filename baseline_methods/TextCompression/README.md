# Compression method calculating cross-entropy

## Prerequisites

***
  - Python 3.6.x


## Run
***
> __A continuación asumimos que el directorio de trabajo es la raíz del repositorio.__  

Para ejecutar el script, usando los datos base de pan14:
  ```sh
  python3 text_compression.py -a "../../corpus/pan14/train.jsonl" \
  -b "../../corpus/pan14/train_truth.jsonl"\
  -c "../../baseline_methods/TextCompression/model/prepare.jsonl" \
  -m "../../baseline_methods/TextCompression/model/model_pan14.joblib"\
  -i "../../corpus/pan14/test.jsonl" \
  -v "../../corpus/pan14/test_truth.jsonl"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan14_pred.jsonl
  ```
Para ejecutar el script, usando los datos base de pan15:
  ```sh
  python3 text_compression.py -a "../../corpus/pan15/train.jsonl" \
  -b "../../corpus/pan15/train_truth.jsonl"\
  -c "../../baseline_methods/TextCompression/model/prepare.jsonl" \
  -m "../../baseline_methods/TextCompression/model/model_pan15.joblib"\
  -i "../../corpus/pan15/test.jsonl" \
  -v "../../corpus/pan15/test_truth.jsonl"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan15_pred.jsonl
  ```

Para ejecutar el script, usando los datos base de pan22:
  ```sh
  python3 text_compression.py -a "../../corpus/pan22/train.jsonl" \
  -b "../../corpus/pan22/train_truth.jsonl"\
  -c "../../baseline_methods/TextCompression/model/prepare.jsonl" \
  -m "../../baseline_methods/TextCompression/model/model_pan22.joblib"\
  -i "../../corpus/pan22/test.jsonl" \
  -v "../../corpus/pan22/test_truth.jsonl"

  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan22_pred.jsonl
  ```

| Args | Description                                                                                                           |
|------|-----------------------------------------------------------------------------------------------------------------------|
| `-a` | train.jsonl file with relative route                                                                                  |
| `-b` | train_truth.jsonl file with relative route                                                                            |
| `-c` | Ruta donde se guardarán las entropias cruzadas                                                             |
| `-i` | test.jsonl file with relative route                                                                                   |
| `-v` | test_truth.jsonl file with relative route                                                                             |
| `-m` | ruta del modelo (de regresión) entrenado a usar para la clasificación de los textos a partir de sus entropías cruzadas |
