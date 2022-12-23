"""
Script containing functions for computing commonly used NLP measures
@author: Shiva Upadhye
@Last modified: 12/23/22
"""

import pandas as pd
import numpy as np
import panphon.distance

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
Compute phonological distance b/w words x and y
"""
def phonDist(x,y):
    dst = panphon.distance.Distance()
    return dst.feature_edit_distance(x,y)

"""
Compute phonological similarity b/w words x and y
"""
def phonSim(x,y):
    dst = panphon.distance.Distance()
    phonDist = dst.feature_edit_distance(x,y)
    num_feat1 = dst.feature_edit_distance(x,"")
    num_feat2 = dst.feature_edit_distance(y,"")
    phonSim = 1 - phonDist/max(num_feat1,num_feat2)
    return phonSim