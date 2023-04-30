from math import *
import csv
import numpy as np
from csv import reader

import pandas as pd

class KNN:
    def loadthefile(file):
        df = pd.read_csv(file, delimiter= '\s+', index_col=False)
        return df

    def normalization(file):
        after = list()
        l = list(file.columns)
        s = list()
        g = list()
        for j in range (len(l)-1):
            after = []
            for i in range(len(file)):
                  x = l[j]
                  normalize = ((file[x][i] - file[x].mean()) / file[x].std() )
                  after.append(normalize)
            s.append(after)
        s.append(file[l[j+1]])
        df = pd.DataFrame(s) 
        df = df.transpose()
        return df
  
    def euclideandistance(train, test):
        distancecalc = 0.0
        for i in range(len(train) - 1):
            distancecalc += (train[i] - test[i]) ** 2
        return sqrt(distancecalc)

    def neighbour(train,test,k):
        allofdist = list()
        for train_row in train:
            dist = KNN.euclideandistance(train_row,test)
            allofdist.append((train_row,dist))
        allofdist.sort(key=lambda dist: dist[1]) 
        neighbors = list()
        for i in range(k):
         neighbors.append(allofdist[i][0])  
        return neighbors
                                                                   
    def tie(train,testrow,k):
        output = []
        neighbors = KNN.neighbour(train, testrow,k) 
        for row in neighbors:
            output.append(row[-1])
        prediction = max(set(output), key = output.count) 
        return prediction

    def accuracy(actual, predicted):
        countofcorrect = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                countofcorrect += 1
        return [countofcorrect / float(len(actual)) * 100.0, countofcorrect]


    def KNNMainProgram(train,test):
        for k in range(1,10):
            actual = []
            predicted = [] 
            print("K = : ", k)
            for i in range(len(test)):
                labelofoutput = KNN.tie(train, test[i], k)
                #print("labeloftest" , test[i][-1],"----> ","predicted" , labelofoutput)
                actual.append(test[i][-1])
                predicted.append(labelofoutput)
            accuracy = KNN.accuracy(actual,predicted)[0]
            numberofinstance = KNN.accuracy(actual,predicted)[1]
            print("number of correctly classified test instances :" , numberofinstance, "--","total number of instances in the test set", 
            len(actual))
            print("Accuracy :", accuracy)
        
                
                 
        


#Main
object = KNN
pendigits_training = object.loadthefile("pendigits_training.txt")
n = object.normalization(pendigits_training)
pendigits_test = object.loadthefile("D:\Machine learning\pendigits_test.txt")
z = object.normalization(pendigits_test)
lol = n.values.tolist()
lol2 = z.values.tolist()
object.KNNMainProgram(lol,lol2)



