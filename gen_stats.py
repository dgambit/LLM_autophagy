

import pandas as pd
from datasets import load_dataset

import spacy


import textdescriptives as td
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textdescriptives/all") 


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import re
import math
import pickle


from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def cleantext(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_words)
    return filtered_text


def TTR(text):
    text = cleantext(text.lower())
    text = re.sub(r'[^a-zA-Z ]', "", text)
    token = text.split(" ")
    token = [t for t in token if (t!="" and t not in stop_words)]
    type = set(token)
    if len(token) !=0:
        TTR = len(type)/len(token)
    else:
        TTR=0
    return TTR


def norm_entropy(text, base=2):
    text = cleantext(text.lower())
    text = re.sub(r'[^a-zA-Z ]', "", text)
    token = text.split(" ")
    token = [t for t in token if (t!="" and t not in stop_words)]
    type = set(token)
    occurs = {t:0 for t in type}
    for t in token:
        occurs[t] +=1
    probs = occurs.values()
    probs = [p/len(token) for p in probs]
    H = -sum([p  * math.log(p) / math.log(base) for p in probs ])
    try:
        H = H/(-sum([1/len(token) * math.log(1/len(token)) / math.log(base) for t in token ]))
    except:
        H = 0
    return H


from collections import Counter

def count_words(word_list):
    return Counter(word_list)


nltk.download('punkt')

def process_text_tokens(text):
    text = str(text)
    stemmer = PorterStemmer()
    text = text.lower()    
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    processed_text = ' '.join(stemmed_words)
    return processed_text


def count_tokens(sent):
    doc = nlp(sent)
    token_dict = {}
    for token in doc:
        try: token_dict[token.pos_].append(process_text_tokens(token))
        except: token_dict[token.pos_] = [process_text_tokens(token)]
    #token_dict["ALL"] = sum(token_dict.values())
    return token_dict



def stats(x):
    doc = nlp(x)
    return doc._.descriptive_stats | doc._.information_theory


def avg(list):
    return(sum(list)/len(list))


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()


n=11

for run in ["I", "II", "III" ,"IV"]:
    
    for i in range(n):
        print(run, i)
        load = load_dataset("danigambit/doc_gen_"+str(i)+"_"+str(run))
        df = pd.DataFrame(load["train"])


        stats_dict = df["doc"].apply(lambda x: stats(x))
        for k in stats_dict[0].keys():
            df[k] = [stats_dict[i][k] for i in range(2050)]


        df["norm_entropy_doc"] = df["doc"].apply(lambda x: norm_entropy(x))
        df["ttr_doc"] = df["doc"].apply(lambda x: TTR(x))
        df["count_tokens"] = df["doc"].apply(lambda x: count_tokens(x))

        tfidf = v.fit_transform(df["doc"])
        df["tf_idf_all"] = [avg(tf) for tf in tfidf.toarray()]
        df["tf_idf_notnnull"] = [avg([i for i in tf if i!=0]) for tf in tfidf.toarray()]

        gen_doc = " ".join(df["doc"])
        entropy_gen = norm_entropy(gen_doc)
        df["entropy_gen"] = [entropy_gen for i in df["id"]]

        ttr_gen = TTR(gen_doc)
        df["ttr_gen"] = [ttr_gen for i in df["id"]]


        file_path = "text_stats/stats_gen_"+str(i)+"_"+run+".pickle"

        print("Saving...")
        with open(file_path, 'wb') as file:
            pickle.dump(df, file)

