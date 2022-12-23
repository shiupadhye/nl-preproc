"""
Script containing functions for computing commonly used NLP measures
@author: Shiva Upadhye
@Last modified: 12/23/22
"""

import pandas as pd
import numpy as np

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

