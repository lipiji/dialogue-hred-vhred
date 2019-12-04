import os
import sys
from utils.bleu import *

pred = []

with open("./pred/res.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        fs = line.split("\t")
        if len(fs) != 3:
            print("error")
            continue
        q, g, r = fs
        g = [w for w in g]
        r = [w for w in r]
        pred.append((g, r))

def get_bleu():
    ma_bleu = 0.
    ma_bleu1 = 0.
    ma_bleu2 = 0.
    ma_bleu3 = 0.
    ma_bleu4 = 0.
    ref_lst = []
    hyp_lst = []
    for g, r in  pred:
        references = g
        hypothesis = r
        ref_lst.append(references)
        hyp_lst.append(hypothesis)
        bleu, precisions, _, _, _, _ = compute_bleu([references], [hypothesis], smooth=False) 
        ma_bleu += bleu
        ma_bleu1 += precisions[0]
        ma_bleu2 += precisions[1]
        ma_bleu3 += precisions[2]
        ma_bleu4 += precisions[3]
    n = len(pred)
    ma_bleu /= n
    ma_bleu1 /= n
    ma_bleu2 /= n
    ma_bleu3 /= n
    ma_bleu4 /= n
    
    mi_bleu, precisions, _, _, _, _ = compute_bleu(ref_lst, hyp_lst, smooth=False)
    mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4 = precisions[0], precisions[1], precisions[2], precisions[3]
    return ma_bleu, ma_bleu1, ma_bleu2, ma_bleu3, ma_bleu4,\
           mi_bleu, mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4


def get_dist():
    unigrams = []
    bigrams = []
    ma_dist1, ma_dist2 = 0., 0.
    for g, r in pred:
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i+1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs))
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)

    n = len(pred)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))

    return ma_dist1, ma_dist2, mi_dist1, mi_dist2


if True:
    ma_bleu, ma_bleu1, ma_bleu2, ma_bleu3, ma_bleu4,\
    mi_bleu, mi_bleu1, mi_bleu2, mi_bleu3, mi_bleu4 = get_bleu()

    ma_dist1, ma_dist2, mi_dist1, mi_dist2 = get_dist()

    print("ma_bleu", ma_bleu)
    print("ma_bleu1", ma_bleu1)
    print("ma_bleu2", ma_bleu2)
    print("ma_bleu3", ma_bleu3)
    print("ma_bleu4", ma_bleu4)
    print("mi_bleu", mi_bleu)
    print("mi_bleu1", mi_bleu1)
    print("mi_bleu2", mi_bleu2)
    print("mi_bleu3", mi_bleu3)
    print("mi_bleu4", mi_bleu4)
    print("ma_dist1", ma_dist1)
    print("ma_dist2", ma_dist2)
    print("mi_dist1", mi_dist1)
    print("mi_dist2", mi_dist2)

    print("& %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" \
            % (ma_bleu*100, ma_bleu1*100, ma_bleu2*100, \
               ma_bleu3*100, ma_bleu4*100, ma_dist1*100, \
               ma_dist2*100, mi_dist1*100, mi_dist2*100))

    print()

