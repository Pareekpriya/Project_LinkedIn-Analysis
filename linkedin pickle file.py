# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:52:28 2023

@author: user
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#loaded the saved model
cosine_similarity= pickle.load(open("C:/Users/user/Downloads/cosine_similarity.pkl", 'rb'))
jobs_df=pickle.load(open("C:/Users/user/Downloads/jobs_df.pkl",'rb'))