"""
Script containing functions for computing commonly used NLP measures
@author: Shiva Upadhye
@Last modified: 12/23/22
"""

import pandas as pd
import numpy as np
import pronouncing
import panphon.distance
import torchtext
from torchtext.vocab import FastText
from torchtext.vocab import GloVe


"""
Get SUBTLEXus word frequencies
"""
def get_subtlexus_freqs(w):
    # load frequency table
    freqs = pd.read_csv("SUBTLEXUS.csv")
    freqs["Word"] = freqs["Word"].str.lower()
    words = freqs["Word"].values
    if w in words:
        wf = freqs[freqs["Word"] == w]["FREQcount"].values[0]
        return wf
    else:
        return 0

"""
Computes phonological distance b/w words x and y
"""
def phonDist(x,y):
    dst = panphon.distance.Distance()
    return dst.feature_edit_distance(x,y)

"""
Computes phonological similarity b/w words x and y
"""
def phonSim(x,y):
    dst = panphon.distance.Distance()
    phonDist = dst.feature_edit_distance(x,y)
    num_feat1 = dst.feature_edit_distance(x,"")
    num_feat2 = dst.feature_edit_distance(y,"")
    phonSim = 1 - phonDist/max(num_feat1,num_feat2)
    return phonSim


"""
Returns the syllabic breakdown of a given word
"""
def syllabify(word):
    return pronouncing.phones_for_word(word)


"""
Computes the semantic distance between two words
"""
def semDistGloVe(w1,w2):
    glove = torchtext.vocab.GloVe(name = '6B', dim = 300)
    x = glove[re.sub(r"['|-]","",w1)].unsqueeze(0)
    y = glove[re.sub(r"['|-]","",w2)].unsqueeze(0)
    semDist = 1-torch.cosine_similarity(x,y)
    return semDist


