# Authorship Verification Using SVC

## Data

***

The data folder contains datasets for train and test in .jsonl format

## Prerequisites

***
  - Using pipenv

  > Make sure `pipenv` is installed. (If not, simply run: `pip install pipenv`.)
  
    ```sh
    # Activate the virtual environment
    pipenv shell
    # Install the dependencies
    pipenv install -r requirements.txt
    ```

## Run

***

> __Below we assume the working directory is the repository root.__  

To run the script, using the pan15 base data:
  ```sh
  python main.py -t "./data/pan15/test.jsonl" -v "./data/pan15/test_truth.jsonl" -n "./data/pan15/train.jsonl"  -y "./data/pan15/train_truth.jsonl" -o "pan15_pred.jsonl"
  ```
Output:
  ```sh
  Predictions saved in:
  ~\BOWSVM\results\pan15.jsonl
  Evaluation saved in:
  ~\BOWSVM\metrics\pan15
  ```

| Args   | Description                                    |
|--------|------------------------------------------------|
| `-t`   | test.jsonl file with relative route            |
| `-v`   | test_truth.jsonl file with relative route      |
| `-n`   | train.jsonl file with relative route           |
| `-y`   | train_truth.jsonl file with relative route     |
| `-o`   | file name of output prediction                 |