# -*- coding: utf-8 -*-

"""
 A baseline authorship verificaion method based on text compression.
 Given two texts text1 and text2 it calculates the cross-entropy of
 text2 using the Prediction by Partical Matching (PPM) compression
 model of text1 and vice-versa.
 Then, the mean and absolute difference of the two cross-entropies
 are used to estimate a score in [0,1] indicating the probability the
 two texts are written by the same author.
 The prediction model is based on logistic regression and can be trained
 using a collection of training cases (pairs of texts by the same or
 different authors).
 Since the verification cases with a score exactly equal to 0.5 are
 considered to be left unanswered, a radius around this value is
 used to determine what range of scores will correspond to the
 predetermined value of 0.5.
 The method is based on the following paper:
     William J. Teahan and David J. Harper. Using compression-based
     language models for text categorization. In Language Modeling
     and Information Retrieval, pp. 141-165, 2003
 The current implementation is based on the code developed in the
 framework of a reproducibility study:
     M. Potthast, et al. Who Wrote the Web? Revisiting Influential
     Author Identification Research Applicable to Information Retrieval.
     In Proc. of the 38th European Conference on IR Research (ECIR 16),
     March 2016.
     https://github.com/pan-webis-de/teahan03
 Questions/comments: stamatatos@aegean.gr
 It can be applied to datasets of PAN-20 cross-domain authorship verification task.
 See details here: http://pan.webis.de/clef20/pan20-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 Usage from command line:
    > python text_compression.py
        -i EVALUATION-FILE
        -o OUTPUT-DIRECTORY
        [-m MODEL-FILE]
 EVALUATION-DIRECTORY (str) is the full path name to a PAN-20 collection
 of verification cases (each case is a pair of texts)
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are
 saved in the PAN-20 format
 Optional parameter:
     MODEL-FILE (str) is the full path name to the trained model
     (default=model_small.joblib, a model already trained on the small
     training dataset released by PAN-20 using logistic regression
     with PPM order = 5)
     RADIUS (float) is the radius around the threshold 0.5 to leave
     verification cases unanswered (dedault = 0.05). All cases with a
     value in [0.5-RADIUS, 0.5+RADIUS] are left unanswered.
 Example:
     > python baseline2.py
     -i "mydata/pan20-authorship-verification-test-corpus.jsonl"
     -o "mydata/pan20-answers"
     -m "mydata/model_small.joblib"
 Additional functions (train_data and train_model) are provided to
 prepare training data and train a new model.
Supplementary files:
    data-small.txt: training data extracted from the small dataset
    provided by PAN-20 authorship verification task
    model.joblib: trained model using logistic regression,
    PPM order=5, using data of data-small.txt
"""

import re
import os
import json
import time
import argparse
import platform
import subprocess
import numpy as np
from pathlib import Path
from math import log
from joblib import dump, load
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuitCV

import sys
sys.path.insert(1,os.path.join(os.path.abspath('.'),"..",".."))
from utils.split import getDataJSON

class Order(object):
    # n - whicht order
    # cnt - character count of this order
    # contexts - Dictionary of contexts in this order
    def __init__(self, n):
        self.n = n
        self.cnt = 0
        self.contexts = {}

    def hasContext(self, context):
        return context in self.contexts

    def addContext(self, context):
        self.contexts[context] = Context()

    def merge(self, o):
        self.cnt += o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                self.contexts[c] = o.contexts[c]
            else:
                self.contexts[c].merge(o.contexts[c])

    def negate(self, o):
        if self.cnt < o.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.contexts[c].negate(o.contexts[c])
        empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
        for c in empty:
            del self.contexts[c]


class Context(object):
    def __init__(self):
        self.chars = {}  # chars - Dictionary containing character counts of the given context
        self.cnt = 0  # cnt - character count of this context

    def hasChar(self, c):
        return c in self.chars

    def addChar(self, c):
        self.chars[c] = 0

    def incCharCount(self, c):
        self.cnt += 1
        self.chars[c] += 1

    def getCharCount(self, c):
        return self.chars[c]

    def merge(self, cont):
        self.cnt += cont.cnt
        for c in cont.chars:
            if not self.hasChar(c):
                self.chars[c] = cont.chars[c]
            else:
                self.chars[c] += cont.chars[c]

    def negate(self, cont):
        if self.cnt < cont.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= cont.cnt
        for c in cont.chars:
            if (not self.hasChar(c)) or (self.chars[c] < cont.chars[c]):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.chars[c] -= cont.chars[c]
        empty = [c for c in self.chars if self.chars[c] == 0]
        for c in empty:
            del self.chars[c]


class Model(object):
    def __init__(self, order, alphSize):
        """
        Parameters
        ----------
        order       List of Order-Objects
        alphSize    Size of the alphabet
        """
        self.cnt = 0  # cnt - count of characters read
        self.alphSize = alphSize
        self.modelOrder = order  # modelOrder - order of the model
        self.orders = []
        for i in range(order + 1):
            self.orders.append(Order(i))

    # print the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printModel(self):
        for i in range(self.modelOrder + 1):
            self.printOrder(i)

    # print a specific order of the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printOrder(self, n):
        o = self.orders[n]
        s = "Order " + str(n) + ": (" + str(o.cnt) + ")\n"
        for cont in o.contexts:
            if n > 0:
                s += "  '" + cont + "': (" + str(o.contexts[cont].cnt) + ")\n"
            for char in o.contexts[cont].chars:
                s += "     '" + char + "': " + \
                    str(o.contexts[cont].chars[char]) + "\n"
        s += "\n"
        print(s)

    # updates the model with a character c in context cont
    def update(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than model order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            order.addContext(cont)
        context = order.contexts[cont]
        if not context.hasChar(c):
            context.addChar(c)
        context.incCharCount(c)
        order.cnt += 1
        if order.n > 0:
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if len(s) == 0:
            return
        for i in range(len(s)):
            if i != 0 and i - self.modelOrder <= 0:
                cont = s[0:i]
            else:
                cont = s[i - self.modelOrder:i]
            self.update(s[i], cont)

    # return the models probability of character c in content cont
    def p(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])
        return float(context.getCharCount(c)) / context.cnt

    # merge this model with another model m, esentially the values for every
    # character in every context are added
    def merge(self, m):
        if self.modelOrder != m.modelOrder:
            raise NameError("Models must have the same order to be merged")
        if self.alphSize != m.alphSize:
            raise NameError("Models must have the same alphabet to be merged")
        self.cnt += m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].merge(m.orders[i])

    # make this model the negation of another model m, presuming that this
    # model was made my merging all models
    def negate(self, m):
        if self.modelOrder != m.modelOrder or self.alphSize != m.alphSize or self.cnt < m.cnt:
            raise NameError("Model does not contain the Model to be negated")
        self.cnt -= m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].negate(m.orders[i])


# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    n = len(s)
    h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]

        h -= log(m.p(s[i], context), 2)
    return h / n


# Calculates the cross-entropy of text2 using the model of text1 and vice-versa
# Returns the mean and the absolute difference of the two cross-entropies
def distance(text1, text2, ppm_order=5):
    mod1 = Model(ppm_order, 256)
    mod1.read(text1)
    d1 = h(mod1, text2)

    mod2 = Model(ppm_order, 256)
    mod2.read(text2)
    d2 = h(mod2, text1)

    mean = (d1 + d2) / 2.0
    l1_distance = abs(d1 - d2)

    return [round(mean, 4), round(l1_distance, 4)]


# ===== Cargar y preparar datos de entrenamiento ===== #
def prepare_data(train, truth, prepared):
    """ Esta función calcula las entropías cruzadas de cada par de textos.
    :param train: Archivo con los datos de entrenamiento (textos)
    :param truth: Archivo con las respuestas del conjunto de entrenanmiento
    :param prepared: Ruta donde se guardarán las entropias cruzadas
    :return:
    """
    
    #with open(truth, 'r') as fp:
    #    labels = []  # List of dictionaries: {'id': , 'same':True/False}
    #    for line in fp:
    #        labels.append(json.loads(line))

    labels = getDataJSON(truth)
        
    with open(train, 'r') as fp:
        tr_data = {}
        data = []
        tr_labels = []
        for i, line in enumerate(fp):
            prob = json.loads(line)  # Diccionario {'id': , 'pair': [text0, text1]}
            prob_id = prob['id']
            pair_0 = prob['pair'][0]
            pair_1 = prob['pair'][1]

            if labels[i]['id'] == prob_id:
                true_label = 1 if labels[i]['value'] else 0
                cross_entropy = distance(pair_0, pair_1)

                data.append(cross_entropy)
                tr_labels.append(true_label)

            #    print(i, prob_id, cross_entropy, labels[i]['same'], true_label)

            #else:
            #    print(f"{labels[i]['id']} no es {prob_id} ")

        # Data for training
        tr_data['data'] = data
        tr_data['labels'] = tr_labels

    with open(prepared, 'w') as fp:
        json.dump(tr_data, fp)


# ===== Entrenamiento de modelo ===== #

def train_model(train_data, path_output_model):
    """ Función para entrenar un modelo de regresión logística.
    :param train_data: Ruta del archivo que contiene las entropías cruzadas (calculadas por la función prepare_data)
    :param path_output_model: Ruta y nombre del modelo entrenado.
    :return:
    """
    with open(train_data) as fp:
        all_data = json.load(fp)  # Dictionary {'data':[[]], 'labels':[]}
        x_train = np.array(all_data['data']).reshape(-1, 2)
        y_train = np.array(all_data['labels'])

        assert x_train.shape == (len(all_data['data']), 2)
        assert y_train.shape == (len(all_data['labels']), )

    model = LogisticRegression(solver='lbfgs', random_state=0, max_iter=500)
    model.fit(x_train, y_train)

    dump(model, path_output_model)


# ===== Predicciones ===== #

def apply_model(path_test,path_truth, path_model, radius=0.05):
    
    BASE_DIR = Path(__file__).resolve().parent
    model = load(path_model)
    print(model)
    answers = []
    with open(path_test, 'r') as fp:
        for i, line in enumerate(fp):
            prob = json.loads(line)
            prob_id = prob['id']
            pair_0 = prob['pair'][0]
            pair_1 = prob['pair'][1]
            cross_entropy = distance(pair_0, pair_1)
            proba_predicted = model.predict_proba([cross_entropy])
            proba_same = proba_predicted[0][1]

            if(path_model.split("/")[-1] != "model_pan15.joblib"):
                if 0.5 - radius <= proba_same <= 0.5 + radius:
                    proba_same = 0.5
            answers.append({'id': prob_id, 'value': round(proba_same, 3)})

            
            if i%50==0:
                print(f"{i+1} elementos analizados")

    folder_name = re.findall(r'.*/([a-zA-Z0-9]+)/.*\.jsonl$',path_test)[0] 
    prediction_folder = os.path.join(BASE_DIR,'prediction',f'{folder_name}_pred.jsonl')

    with open(prediction_folder, 'w+') as f:
        for ans in answers:
            json.dump(ans, f)
            f.write('\n')
        print("Predictions saved in:", prediction_folder)
    
    time.sleep(4)
    EVALUATION_DIR = os.path.join(BASE_DIR,"..","..","resultados","TextCompression")    
    evaluation_route = os.path.join(EVALUATION_DIR,folder_name)
    if not os.path.exists(evaluation_route):
        os.makedirs(evaluation_route)

    if "Windows" in platform.system():
        subprocess.run(["python","../../utils/verif_evaluator.py","-i",
            path_truth,"-a",prediction_folder,"-o", evaluation_route])
    else:
        subprocess.run(["python3","../../utils/verif_evaluator.py","-i",
            path_truth,"-a",prediction_folder,"-o", evaluation_route])

def main():
    parser = argparse.ArgumentParser(description='PAN22 Cross-domain Authorship Verification task: Baseline Compressor')
    parser.add_argument('-a', type=str, help='Full path name to the train dataset JSONL file')
    parser.add_argument('-b', type=str, help='Full path name to the evaluation train_truth dataset JSONL file')
    parser.add_argument('-c', type=str, help='Full path name to the where the crossed entropies will be saved JSONL file')
    parser.add_argument('-m', type=str, default='model_small.joblib', help='Full path name to the model file')
    parser.add_argument('-i', type=str, help='Full path name to the evaluation dataset JSONL file')
    parser.add_argument('-v', type=str, help='Full path name to the evaluation truth dataset JSONL file')
    parser.add_argument('-r', type=float, default=0.05, help='Radius around 0.5 to leave verification cases unanswered')
    args = parser.parse_args()

    if not args.a:
        print('ERROR: The evaluation file is required')
        parser.exit(1)
    if not args.b:
        print('ERROR: The evaluation file is required')
        parser.exit(1)
    if not args.c:
        print('ERROR: The evaluation file is required')
        parser.exit(1)
    if not args.i:
        print('ERROR: The evaluation file is required')
        parser.exit(1)
    if not args.v:
        print('ERROR: The evaluation truth file is required')
        parser.exit(1)
    if ".jsonl" not in args.i:
        raise ValueError('The evaluation file will be .jsonl')
    if ".jsonl" not in args.v:
        raise ValueError('The evaluation truth file will be .jsonl')

    print('Exec prepare_data')
    prepare_data(args.a, args.b, args.c)
    print('Exec train_model')
    train_model(args.c, args.m)
    print('Exec apply_model')
    apply_model(args.i, args.v, args.m, args.r)


if __name__ == '__main__':
    main()