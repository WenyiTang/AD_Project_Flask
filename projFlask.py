from flask import Flask, request
import json
import ast
import sys
import pandas as pd
import numpy as np
import pickle
import nltk

from sklearn.metrics import r2_score
import WmaModel
import recModel
import nltkmodules
import requests
application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello World!"

@application.route("/suggestnextmeal", methods = ["POST"])
def suggestNextMeal() :

    WmaModel.wmaModel()
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

@application.route("/getData", methods = ["POST"])
def getData():

    recModel.recmodel()
    modelFile = open("recmodel", "rb")
    model = pickle.load(modelFile)

    jsonObject = request.get_json(force=True)
    jsonString = json.dumps(jsonObject)
    data = json.loads(jsonString)
    id = data["id"]
    title = data["title"]
    description = data["description"]
    feeling = data["feeling"]
    track_score = data["track_score"]
    input = data["input"]
    feel = data["feel"]
    track = data["track"]

    s = [input, feel, track]
    #print(s)
    d = {"id" : id, "title" : title, "description" :description, "feeling" : feeling, "track_score" : track_score}
    df = pd.DataFrame(data=d)
    #print(df.head())
    
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

if __name__ == "__main__":
    application.run()