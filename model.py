import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats as ss
import numpy as np
from os import path

def ranking():
    # This function computes the rankings without starring. It uses tf-idf
    # to determine the similarity between the profiles and the given prompt.
    # Then a score based on the number of connects is calculated. Adding the
    # together will give a score.
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "potential-talents - Aspiring human resources - seeking human resources.csv")
    a = pd.read_csv(filepath)
    b = a["job_title"].str.replace('HR', 'Human Resources', regex=True)
    c = []
    for i in range(len(b)):
        c.append(b[i])
    c.append("Aspiring human resources")
    c.append("seeking human resources")
    tfidf = vect.fit_transform(c)
    pairwise_similarity = tfidf * tfidf.T
    r = pairwise_similarity[-2:, :].toarray()
    r1 = r[0]
    r2 = r[1] 
    n_applicants = len(r1)-2
    r1 = r1[:n_applicants]
    r2 = r2[:n_applicants]
    r3 = r1+r2
    d = a["connection"].str.replace('500+ ', '500', regex=False)
    d = d.astype("int")
    e = []
    for i in range(len(d)):
        if d[i] == 500:
            e.append(len(d))
        else:
            e.append(d[i]/500*len(d))
    e1 = ss.rankdata(r3)
    final_r = e1+e
    e2 = ss.rankdata(final_r)

    return final_r 
  
       


def star(starred, score):
    # This function computes the rankings with starring. After starring, the
    # starred profiles are assigned the highest score, a similar procedure 
    # to above will be perforrmed to determine the remaining rankings.
    
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "potential-talents - Aspiring human resources - seeking human resources.csv")
    a = pd.read_csv(filepath)
    b = a["job_title"].str.replace('HR', 'Human Resources', regex=True)
    c = []
    for i in range(len(b)):
        c.append(b[i])
    c.append("Aspiring human resources")
    c.append("seeking human resources")
    tfidf = vect.fit_transform(c)
    pairwise_similarity = tfidf * tfidf.T
    r = pairwise_similarity[-2:, :].toarray()
    r1 = r[0]
    r2 = r[1] 
    n_applicants = len(r1)-2
    r1 = r1[:n_applicants]
    r2 = r2[:n_applicants]
    r3 = r1+r2
    d = a["connection"].str.replace('500+ ', '500', regex=False)
    d = d.astype("int")
    e = []
    for i in range(len(d)):
        if d[i] == 500:
            e.append(len(d))
        else:
            e.append(d[i]/500*len(d))
    e1 = ss.rankdata(r3)
    final_r = e1+e
    e2 = ss.rankdata(final_r)
    v1 = pairwise_similarity[starred,:].toarray()
    v2 = ss.rankdata(v1)
    final_r1 = v2+e
    score = final_r1
    score[starred] = n_applicants*2

    return score
    