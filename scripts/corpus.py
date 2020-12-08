import json
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def statistic():
    corpus_dict = []
    with open("../corpus/lda_tagged_all_articles.json",'r') as load_f:
        for row in load_f:
            corpus_dict.append(json.loads(row))
    
    string = ""
    y_count = 0
    max_length = 0
    long_count = 0
    for passage in corpus_dict:
        string += passage['content']
        if "10" in passage['index']: 
            y_count +=1
            if len(passage['content']) > 600 : long_count += 1
            if len(passage['content']) > max_length: max_length = len(passage['content'])
    
    print(len(string))
    print(y_count)
    print(long_count)
    print(max_length)

def generate_pairdata():
    argu_corpus = []
    with open("../corpus/lda_tagged_all_articles.json",'r') as load_f:
        for row in load_f:
            row_dict = json.loads(row)
            if "10" in row_dict['index']:
                argu_corpus.append(row_dict['content'])
    
    argu_sentence = []
    sentence_count = []
    for passage in argu_corpus:
        for paragraph in passage.split("\n"):
            if len(re.split("。|？|”|“|！", paragraph))-1 > 0: 
                sentence_count.append(len(re.split("。|？|”|“|！", paragraph))-1)
    df = pd.Series(sentence_count, name = 'count')
    df = pd.DataFrame(df)
    
    print(df.describe())

    counts = np.bincount(sentence_count)
    print(counts)

generate_pairdata()
print(matplotlib.pyplot.get_backend())
 

