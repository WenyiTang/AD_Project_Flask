from flask import Flask, request, jsonify
import json
import ast
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score
import recModel
import WmaModel
import requests

app = Flask(__name__)

@app.route("/enterquery", methods = ["POST", "GET"])
def searchQuery() :
    query_url = "http://localhost:8080/api/recommend/enterquery1"
    response = requests.get(query_url)
    print(response.json())
    full_query = response.json()
    s = []
    s.append(full_query['input'])
    s.append(full_query['feeling'])
    s.append(full_query['track'])
    print(s)

    modelFile = open("recmodel", "rb")
    model = pickle.load(modelFile)
    df = getData()
    result = model.search(s, df)
    print(result)
    result_dict = {
    "res0" : -1,
    "res1" : -1,
    "res2" : -1,
    "res3" : -1,
    "res4" : -1,
    "goodResult" : ""
    }
    for x in range(5):
        index = "res" + str(x)
        result_dict[index] = int(result[x])
    result_dict["goodResult"] = result[-1]
    return result_dict

def getData():
    data_url = "http://localhost:8080/api/recommend/passData"
    data_response = requests.get(data_url)
    data = data_response.json()
    df = pd.DataFrame()
    df['id'] = data['id']
    df['title'] = data['title']
    df['description'] = data['description']
    df['feeling'] = data['feeling']
    df['track_score'] = data['track_score']
    print(df.head())
    return df

@app.route("/suggestnextmeal", methods = ["POST"])
def suggestNextMeal() :
    modelFile = open("wmaModel", "rb")
    model = pickle.load(modelFile)
    jsonObject = request.get_json(force=True)
    jsonString = json.dumps(jsonObject)
    data = json.loads(jsonString)
    print(data, file=sys.stdout)
    targetCount = int(data["targetCount"])
    trackScore = ast.literal_eval(data["trackScore"])
    trackScoreDf = pd.DataFrame(trackScore)
    trackScoreDf.index = trackScoreDf.index + 1
    result = model.isConsistent(trackScoreDf, targetCount)
    return result

if __name__ == "__main__":
    recModel.recmodel()
    WmaModel.wmaModel()
    app.run(debug = True)