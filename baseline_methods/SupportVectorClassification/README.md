# Authorship Verification Using SVC

## Run

***
> __Below we assume the working directory is the repository root.__  

To run the script, using the pan14 base data:
  ```sh
  python3 main.py -t "../../corpus/pan14/test.jsonl" \
  -v "../../corpus/pan14/test_truth.jsonl" \
  -n "../../corpus/pan14/train.jsonl" \
  -y "../../corpus/pan14/train_truth.jsonl" \
  -o "pan14_pred.jsonl"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/SupportVectorClassification/prediction/pan14_pred.jsonl
  Evaluation saved in:
  ~/AuthorshipVerification-Baseline/evaluationSupportVectorClassification/pan14/
  ```


| Args   | Description                                    |
|--------|------------------------------------------------|
| `-t`   | test.jsonl file with relative route            |
| `-v`   | test_truth.jsonl file with relative route      |
| `-n`   | train.jsonl file with relative route           |
| `-y`   | train_truth.jsonl file with relative route     |
| `-o`   | file name of output prediction                 |