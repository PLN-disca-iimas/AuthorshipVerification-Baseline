import os
import re
import platform
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
try:
    from ...utils.split import getDataJSON
except:
    import sys
    sys.path.insert(1,os.path.join(os.path.abspath('.'),"..",".."))
    from utils.split import getDataJSON


def svcBinaryClassifier():
    parser = argparse.ArgumentParser(description='SVC script AA@PAN')
    parser.add_argument('-t', type=str,
        help='Path to the jsonl-file with the text test dataset')
    parser.add_argument('-v', type=str,
        help='Path to the jsonl-file with ground test truth scores')
    parser.add_argument('-n', type=str,
        help='Path to the jsonl-file with the text train dataset')
    parser.add_argument('-y', type=str,
        help='Path to the jsonl-file with ground train truth scores')
    parser.add_argument('-o', type=str,
        help='output files name')
    args = parser.parse_args()

    if not args.v:
        raise ValueError('The ground test truth path is required')
    if not args.t:
        raise ValueError('The test dataset is required')
    if not args.y:
        raise ValueError('The ground truth train path is required')
    if not args.n:
        raise ValueError('The train dataset is required')
    if not args.o:
        raise ValueError('The output folder path is required')
    elif ".jsonl" not in args.o:
        raise ValueError('The output format will be .jsonl')

    data = pd.DataFrame(getDataJSON(args.n)).set_index("id")
    data[['text1','text2']] = pd.DataFrame(data.pair.tolist(), index= data.index)
    del data["pair"]
    data2 = pd.DataFrame(getDataJSON(args.y)).set_index("id")
    data = pd.merge(data,data2,how='outer',left_index=True,right_index=True)
    del data2

    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    X_2 = unigram_vectorizer.fit_transform(data["text1"])
    X_2 = X_2.toarray() - unigram_vectorizer.transform(data["text2"]).toarray()

    #Entrenamiento SVC
    clf = SVC(gamma='auto')
    clf.fit(X_2, data["value"])

    data_test = pd.DataFrame(getDataJSON(args.t)).set_index("id")
    data_test[['text1','text2']] = pd.DataFrame(data_test.pair.tolist(), index= data_test.index)
    del data_test["pair"]
    data2 = pd.DataFrame(getDataJSON(args.v)).set_index("id")
    data_test = pd.merge(data_test,data2,how='outer',left_index=True,right_index=True)
    del data2

    #Predicción
    X_test = unigram_vectorizer.transform(data_test["text1"]).toarray() -\
            unigram_vectorizer.transform(data_test["text2"]).toarray()
    y_pred = clf.predict(X_test)

    BASE_DIR = Path(__file__).resolve().parent
    with open(os.path.join(BASE_DIR,"prediction",args.o),"w+") as f:
        for x,y in zip(data_test.index,y_pred):
            f.write(str({"id":x, "value":int(y)}).replace("'",'"')+"\n")
        print("Predictions saved in:")
        print(os.path.join(BASE_DIR,"prediction",args.o))
    
    EVALUATION_DIR = os.path.join(BASE_DIR,"..","..","resultados","SupportVectorClassification")    
    evaluation_route = os.path.join(EVALUATION_DIR,re.findall(r'^([A-Za-z0-9]+).*\.jsonl$',args.o)[0])
    if not os.path.exists(evaluation_route):
        os.makedirs(evaluation_route)
    
    if "Windows" in platform.system():
        subprocess.run(["python","../../utils/verif_evaluator.py","-i",
            args.v,"-a",os.path.join(BASE_DIR,"prediction",args.o),"-o",
            evaluation_route], capture_output=True)
    else:
        subprocess.run(["python3","../../utils/verif_evaluator.py","-i",
            args.v,"-a",os.path.join(BASE_DIR,"prediction",args.o),"-o",
            evaluation_route], capture_output=True)
    
    print("Evaluation saved in:")
    print(evaluation_route)
if __name__ == "__main__":
    svcBinaryClassifier()