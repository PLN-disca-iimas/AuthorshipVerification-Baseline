#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import platform
import subprocess
import random
import re
import os
import glob
import shutil
from itertools import combinations

import numpy as np
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from seaborn import kdeplot
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from ...utils.split import getDataJSON
    from ...utils.verif_evaluator import evaluate_all
except:
    import sys
    sys.path.insert(1,os.path.join(os.path.abspath('.'),"..",".."))
    from utils.split import getDataJSON
    from utils.verif_evaluator import evaluate_all


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rescale(value, orig_min, orig_max, new_min, new_max):
    """
    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.
    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.
    Returns
    ----------
    new_value: float
        The rescaled value.
    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)


def correct_scores(scores, p1, p2):
    new_scores = []
    for sc in scores:
        if sc <= p1:
            sc = rescale(sc, 0, p1, 0, 0.49)
            new_scores.append(sc)
        elif sc > p1 and sc < p2:
            new_scores.append(0.5)
        else:
            sc = rescale(sc, p2, 1, 0.51, 1)
            new_scores.append(sc)
    return np.array(new_scores)


def main():
    parser = argparse.ArgumentParser(description='Distance-based verification: PAN20 baseline')

    # data settings:
    parser.add_argument('-input_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the input pairs')
    parser.add_argument('-input_truth', type=str, required=True,
                        help='Path to the ground truth-file for the input pairs')
    parser.add_argument('-test_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the test pairs')
    parser.add_argument('-test_truth', type=str, required=True,
                        help='Path to the ground truth-file for the test pairs')
    parser.add_argument('-output', type=str, required=True,
                        help='Path to the output folder for the predictions.\
                             (Will be overwritten if it exist already.)')

    # algorithmic settings:
    parser.add_argument('-seed', default=2020, type=int,
                        help='Random seed')
    parser.add_argument('-vocab_size', default=3000, type=int,
                        help='Maximum number of vocabulary items in feature space')
    parser.add_argument('-ngram_size', default=4, type=int,
                        help='Size of the ngrams')
    parser.add_argument('-num_iterations', default=0, type=int,
                        help='Number of iterations (`k`); zero by default')
    parser.add_argument('-dropout', default=.5, type=float,
                        help='Proportion of features to keep in each iteration')

    args = parser.parse_args()
    print('args: ', args)

    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        shutil.rmtree('prediction')
    except FileNotFoundError:
        pass
    os.mkdir('prediction')

    gold = {}
    for line in getDataJSON(args.input_truth):        
        gold[line['id']] = int(line['value'])

    # truncation for development purposes
    cutoff = 0
    if cutoff:
        gold = dict(random.sample(gold.items(), cutoff))
        print(len(gold))

    texts = []
    i=0
    for line in tqdm(open(args.input_pairs, encoding="utf8")):
        d = json.loads(line.strip())
        if d['id'] in gold:
            texts.extend(d['pair'])
        i+=1
    
#        if i==100000:
#            break

    print(len(texts))

    print('-> constructing vectorizer')
    vectorizer = TfidfVectorizer(max_features=args.vocab_size, analyzer='char',
                                 ngram_range=(args.ngram_size, args.ngram_size))
    vectorizer.fit(texts)

    #vectorizer = HashingVectorizer(n_features=args.vocab_size, analyzer='char',
      #                           ngram_range=(args.ngram_size, args.ngram_size))
    #vectorizer.fit(texts)

    print('HECHO')

    if args.num_iterations:
        total_feats = len(vectorizer.get_feature_names())
        keep_feats = int(total_feats * args.dropout)

        rnd_feature_idxs = []
        for _ in range(args.num_iterations):
            print(_)
            rnd_feature_idxs.append(np.random.choice(total_feats,
                                                     keep_feats,
                                                     replace=False))
        rnd_feature_idxs: ndarray = np.array(rnd_feature_idxs)

    print('-> calculating pairwise similarities')
    similarities, labels = [], []
    for line in tqdm(open(args.input_pairs, encoding="utf8")):
        d = json.loads(line.strip())
        if d['id'] in gold:
            x1, x2 = vectorizer.transform(d['pair']).toarray()
            if args.num_iterations:
                similarities_ = []
                for i in range(args.num_iterations):
                    similarities_.append(cosine_sim(x1[rnd_feature_idxs[i, :]],
                                                    x2[rnd_feature_idxs[i, :]]))
                similarities.append(np.mean(similarities_))
            else:
                similarities.append(cosine_sim(x1, x2))
            labels.append(gold[d['id']])
    print('HECHO')

    similarities = np.array(similarities, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    kdeplot(similarities, label='orig cos sim')

    print('-> grid search p1/p2:')
    step_size = 0.01
    thresholds = np.arange(0.01, 0.99, step_size)
    combs = [(p1, p2) for (p1, p2) in combinations(thresholds, 2) if p1 < p2]

    params = {}
    for p1, p2 in tqdm(combs):
        corrected_scores = correct_scores(similarities, p1=p1, p2=p2)
        
        score = evaluate_all(pred_y=corrected_scores,
                             true_y=labels)

        params[(p1, p2)] = score['overall']
    opt_p1, opt_p2 = max(params, key=params.get)
    print('optimal p1/p2:', opt_p1, opt_p2)
    plt.axvline(opt_p1, ls='--', c='darkgrey')
    plt.axvline(opt_p2, ls='--', c='darkgrey')

    corrected_scores = correct_scores(similarities, p1=opt_p1, p2=opt_p2)
    print('optimal score:', evaluate_all(pred_y=corrected_scores,
                                         true_y=labels))


    kdeplot(corrected_scores, label='corr cos sim')
    corr_p1, corr_p2 = correct_scores([opt_p1, opt_p2], p1=opt_p1, p2=opt_p2)
    plt.axvline(corr_p1, ls='--', c='darkgrey')
    plt.axvline(corr_p2, ls='--', c='darkgrey')
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.savefig('kde.pdf')
    plt.clf()

    print('-> determining optimal threshold')
    scores = []
    for th in np.linspace(0.25, 0.75, 1000):
        #print(th)
        adjusted = (corrected_scores >= th) * 1
        scores.append((th,
                       f1_score(labels, adjusted),
                       precision_score(labels, adjusted),
                       recall_score(labels, adjusted)))
    thresholds, f1s, precisions, recalls = zip(*scores)

    max_idx = np.array(f1s).argmax()
    max_f1 = f1s[max_idx]
    max_th = thresholds[max_idx]

    print(f'Dev results -> F1={max_f1} at th={max_th}')
    print('comenzando con los plot')

    plt.plot(thresholds, precisions, label='precision')
    plt.plot(thresholds, recalls, label='recall')
    plt.plot(thresholds, f1s, label='F1')
    plt.axvline(max_th, ls='-', c='darkgrey')
    plt.xlim([0, 1])
    plt.gca().set_xlabel('theta')
    plt.gca().legend()
    plt.gca().set_facecolor('lightgrey')
    plt.title(f'f1={round(max_f1, 4)} @ theta={round(max_th, 4)}')
    plt.tight_layout()
    plt.savefig('dev_precrec.pdf')

    print('se creo pdf')

    print('-> calculating test similarities')
    with open('prediction' + os.sep + args.output, 'w') as outf:
        for line in tqdm(open(args.test_pairs, encoding="utf8")):
            d = json.loads(line.strip())
            problem_id = d['id']
            x1, x2 = vectorizer.transform(d['pair']).toarray()
            if args.num_iterations:
                similarities_ = []
                for i in range(args.num_iterations):
                    similarities_.append(cosine_sim(x1[rnd_feature_idxs[i, :]],
                                                    x2[rnd_feature_idxs[i, :]]))
                    similarity = np.mean(similarities_)
            else:
                similarity = cosine_sim(x1, x2)

            similarity = correct_scores([similarity], p1=opt_p1, p2=opt_p2)[0]
            r = {'id': problem_id, 'value': similarity}
            outf.write(json.dumps(r) + '\n')


    BASE_DIR = Path(__file__).resolve().parent
    EVALUATION_DIR = os.path.join(BASE_DIR,"..","..","resultados","CosineSimilarity")    
    evaluation_route = os.path.join(EVALUATION_DIR,re.findall(r'^([A-Za-z0-9]+).*\.jsonl$',args.output)[0])
        
    if not os.path.exists(evaluation_route):
        os.makedirs(evaluation_route)

    if "Windows" in platform.system():
        subprocess.run(["python","../../utils/verif_evaluator.py","-i",
            args.test_truth,"-a",os.path.join(BASE_DIR,"prediction",args.output),"-o",
            evaluation_route], capture_output=True)
    else:
        subprocess.run(["python3","../../utils/verif_evaluator.py","-i",
            args.test_truth,"-a",os.path.join(BASE_DIR,"prediction",args.output),"-o",
            evaluation_route], capture_output=True)            


if __name__ == '__main__':
    main()