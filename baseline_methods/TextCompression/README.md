# Compression method calculating cross-entropy

## Prerequisites

***
  - Python 3.6.x


## Run
***
> __Below we assume the working directory is the repository root.__  

To run the script, using the pan14 base data:
  ```sh
  python3 text_compression.py -i "../../corpus/pan14/test.jsonl" \
  -v "../../corpus/pan14/test_truth.jsonl" \
  -m "./model/model_small.joblib"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan14_pred.jsonl
  ```


| Args   | Description                                    |
|--------|------------------------------------------------|
| `-i`   | test.jsonl file with relative route            |
| `-v`   | test_truth.jsonl file with relative route      |
| `-m`   | ruta del modelo (de regresión) entrenado a usar para la clasificación de los textos a partir de sus entropías cruzadas |
