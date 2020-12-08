import json
import re
import pandas as pd
import numpy as np

def generate_pairdata():
    argu_corpus = []
    with open("../corpus/lda_tagged_all_articles.json",'r') as load_f:
        for row in load_f:
            row_dict = json.loads(row)
            if "10" in row_dict['index']:
                argu_corpus.append(row_dict['content'])
    
    sentence_pair = []
    for passage in argu_corpus:
        for paragraph in passage.split("\n"):
            sen_list = re.split("。|？|”|“|！", paragraph)
            if len(sen_list)-1 > 1: 
                sentence_pair.append(sen_list[0:2])
    for pair in sentence_pair:
        print(pair[0] + " " + pair[1])


generate_pairdata()
                