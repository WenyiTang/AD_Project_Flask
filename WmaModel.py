import pickle
import numpy as np

class WmaModel :

    def __init__(self) : pass

    def isConsistent(self, df, nConsecutive) :
        sum = 0
        for i in np.arange(1, nConsecutive + 1) :
            sum += i

        weights = []
        for val in np.arange(1, nConsecutive + 1) :
            weight = val / sum
            weights.append(weight)
        
        maxWMA = 0
        for weight in weights :
            maxWMA += weight

        result = df[0].rolling(nConsecutive).apply(lambda x : np.sum(weights * x))
        currentWMA = result.iloc[-1]

        if currentWMA == maxWMA :
            return "true"
        else :
            return "false"

def wmaModel():
    wmaModel_ = WmaModel()
    modelFile = open("wmaModel", "wb")
    pickle.dump(wmaModel_, modelFile)