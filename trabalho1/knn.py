import numpy as np
import operator
import collections
import pandas
from random import randrange
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

norm_p = lambda x,y,p : abs((x-y))**p
euclidian = lambda list1,list2: np.sqrt(sum(map(norm_p,list1,list2,[2]*len(list1))))
manhatan = lambda list1,list2: sum(map(norm_p,list1,list2,[1]*len(list1)))

metrics = {
    "euclidian" : euclidian,
    "manhatan" : manhatan
}

class KNNClassifier:
    def __init__(self, metric="euclidian", n_neighbors=3):
        if metric not in metrics:
            message = "invalid metric. the acceptable values are :"
            for k in metrics.keys():
                message += " " + k
            raise Exception(message)
        self.n_neighbors = n_neighbors
        self.metric_name = metric
        self.metric_func = metrics[metric]
    def fit(self, x_train,y_train):
        if len(x_train) != len(y_train):
            raise Exception("the size of inputs must be equals")
        self.x_train = x_train
        self.y_train = y_train
    def predict(self,x_test):
        distances = []
        result = []
        for test in x_test:
            for index in range(len(self.x_train)):
                distance = self.metric_func(self.x_train[index],test)
                distances.append((self.y_train[index], distance))
            distances = sorted(distances, key = lambda tup : tup[1])
            classes = collections.Counter(map(lambda x : x[0], distances[:self.n_neighbors]))
            clas = classes.most_common(1)
            result.append(clas[0][0])
            distances.clear()
        return result

data = pandas.read_csv("./data/dermatology.csv", header=None)
data.info()

def calc_median(array):
    values = []
    for v in array:
        if v != '?':
            v = int(v)
            values.append(v)
    return np.median(values)

median = calc_median(data.iloc[:,33])
data.iloc[:,33] = list(map(lambda value: median if value == '?' else value, data.iloc[:,33]))
data.iloc[:,33] = data.iloc[:,33].astype(np.int64)

atribbutes = data.iloc[:,:34]
target = data.iloc[:,34]

def train_test_split(data,target,size=0.3):
    test_size = int(len(target)*size)
    numbers = []
    x_train,y_train,x_test,y_test = [],[],[],[]
    while len(numbers) != test_size:
        v = randrange(len(target))
        if v not in numbers:
            numbers.append(v)
            x_test.append(data.iloc[v,:].values)
            y_test.append(target[v])
    for i in range(len(data)):
        if i not in numbers:
            x_train.append(data.iloc[i,:].values)
            y_train.append(target[i])
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = train_test_split(atribbutes,target,0.3)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
