from dataclasses import dataclass
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class recModel :
    def __init__(self) : pass
    def search(self, q, data):
        df, df_combine_only, combine_arr = prepDataBase(data)
        docs_clean = cleanData(combine_arr)
        tfidf, feature_vectors, vocab = prepDBModel(docs_clean)
        query_clean = cleanGetQuery(q)
        query_feature_vector = prepQueryModel(tfidf, query_clean, vocab)
        sorted_series, goodResult = getSimilarity(query_feature_vector, feature_vectors, df, q)
        resultFoodEntryId = getResult(sorted_series, df_combine_only, combine_arr)
        resultFoodEntryId.append(goodResult)
        return resultFoodEntryId

def recmodel():
    recModel_ = recModel()
    modelFile = open("recmodel", "wb")
    pickle.dump(recModel_, modelFile)

def prepDataBase(data):
    df = data
    #df = pd.read_csv('formdata.csv')
    def rename_ontrack(x):
        if (x == 1):
            return "ontrack"
        elif (x == 0):
            return "offtrack"
    df["track_score"] = df["track_score"].apply(rename_ontrack)
    df_combine = df.copy()
    df_combine["combine"] = df_combine["title"] + ", " + df_combine["description"] + ", " + df_combine["feeling"]+ ", " + df_combine["track_score"]
    df_combine_only = df_combine.loc[:, ["id", "combine"]]
    combine_arr = df_combine_only["combine"].values
    return df, df_combine_only, combine_arr

def preprocess(docs):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    docs_clean = []
    punc = str.maketrans("", "", string.punctuation)
    for doc in docs:
        doc_no_punc = doc.translate(punc)
        words = doc_no_punc.lower().split()
        words = [lemmatizer.lemmatize(word, 'v')
                for word in words if word not in stop_words]
        docs_clean.append(' '.join(words))

    return docs_clean

def cleanData(combine_arr):
    docs_clean = preprocess(combine_arr)
    return docs_clean

def prepDBModel(docs_clean):
    tfidf = TfidfVectorizer()
    feature_vectors = tfidf.fit_transform(docs_clean).toarray()
    vocab = tfidf.get_feature_names()
    df_vocab = pd.DataFrame(data=feature_vectors,
                    columns=vocab)
    return tfidf, feature_vectors, vocab

def cleanGetQuery(q):
    query = [q[0]]
    query_clean = preprocess(query)
    if ("cry" in query_clean[0] or "pensive" in query_clean[0]):
        query_clean[0] = query_clean[0].replace("cry", "")
        query_clean[0] = query_clean[0].replace("pensive", "")
        query_clean[0] = query_clean[0] + " happy joy"
    print(query_clean)
    return query_clean

def prepQueryModel(tfidf, query_clean, vocab):
    query_feature_vector = tfidf.transform(query_clean).toarray()
    query_df = pd.DataFrame(data=query_feature_vector,
                        columns=vocab)
    return query_feature_vector

def getSimilarity(query_feature_vector, feature_vectors, df, q):
    similarity = cosine_similarity(query_feature_vector, feature_vectors)
    query_similarity = similarity[0]
    goodResults = "false"
    
    if (query_similarity.max() > 0.3 and len(df) >= 30):
        goodResults = "true"
    print(goodResults)

    #if based on inputs, things are similar enough, add weights to feeling and track_score
    if (goodResults == "true"):
        print("adding weights")
        #score +0.1 if feeling matches, +0.1 if track_score matches
        for x in range(len(query_similarity)):
            feel = df.loc[x,'feeling']
            track = df.loc[x,'track_score']
            if (track == q[2]):
                query_similarity[x] += 0.1

            if ((q[1] != "CRY" or q[1] != "PENSIVE") and feel == q[1]):
                query_similarity[x] += 0.1
            elif (q[1] == "CRY" and feel == "JOY"):
                query_similarity[x] += 0.1
            elif (q[1] == "PENSIVE" and feel == "HAPPY"):
                query_similarity[x] += 0.1

        series = pd.Series(query_similarity, index=df.index)
        sorted_series = series.sort_values(ascending=False)

        #shuffle top 10 and get 5
        sorted_series_10 = sorted_series[sorted_series > 0.3].head(10)
        #if dont have at least 5 scores > 0.3, result is not good
        if (len(sorted_series_10) < 5):
            goodResults = "false"
            sorted_series_out = sorted_series[sorted_series != 0].head(5).sort_values(ascending=False)
        else:
            sorted_series_shuff = sorted_series_10.sample(frac=1)
            sorted_series_out = sorted_series_shuff.head(5).sort_values(ascending=False)

    else:
        series = pd.Series(query_similarity, index=df.index)
        sorted_series = series.sort_values(ascending=False)
        sorted_series_nozero = sorted_series[sorted_series != 0]

        sorted_series_10 = sorted_series_nozero.head(10)
        sorted_series_shuff = sorted_series_10.sample(frac=1)
        sorted_series_out = sorted_series_shuff.head().sort_values(ascending=False)

    return sorted_series_out, goodResults

def getResult(sorted_series, df_combine_only, combine_arr):
    resultFoodEntryId = []
    print("Search results: \n")
    for index in sorted_series.index:
        resultFoodEntryId.append(df_combine_only.loc[index, "id"])
        print(combine_arr[index], " [score = ", sorted_series[index], "] ", "[FoodEntryId = ", df_combine_only.loc[index, "id"], "]", sep=" ")
    return resultFoodEntryId