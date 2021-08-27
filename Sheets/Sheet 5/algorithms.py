import itertools
import math
from collections import Counter
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np
import random


class DecisionTree:
    def __init__(self, gain, _n=5, _pi=0.95):
        self.gain = gain
        self._n = _n
        self._pi = _pi
        self.tree = {}
        self.data = {}

    def getName(self):
        return "DecisionTree"

    def train(self, X, Y, parent=None):
        if not self.data:
            self.data = {"X": X, "Y": Y}
        if(parent is None):
            parent = self.tree
        n = len(X)
        ni = dict(Counter(Y))
        ni_n = {k: v/n for k, v in ni.items()}
        purity = max(ni_n.values())
        if n <= self._n or purity >= self._pi:
            c_ = max(ni_n, key=ni_n.get)
            parent["max"] = c_
            parent["table"] = ni
            return
        splitPoint, bestScore = None, 0
        for attrX in range(len(X[0])):
            if isinstance(X[0][attrX], Number):
                v, score = self.evaluateNumericAttribute(X, Y, attrX)
                if score > bestScore:
                    splitPoint, bestScore = (attrX, v), score
            else:
                v, score = self.evaluateCategoricalAttribute(X, Y, attrX)
                if score > bestScore:
                    splitPoint, bestScore = (attrX, v), score
        if splitPoint is not None:
            if isinstance(X[0][splitPoint[0]], Number):
                Dy = [indx for indx, x in enumerate(
                    X) if x[splitPoint[0]] <= splitPoint[1]]
                Dn = [indx for indx, x in enumerate(
                    X) if not x[splitPoint[0]] <= splitPoint[1]]
            else:
                Dy = [indx for indx, x in enumerate(
                    X) if x[splitPoint[0]] in splitPoint[1]]
                Dn = [indx for indx, x in enumerate(
                    X) if x[splitPoint[0]] not in splitPoint[1]]

            parent[str(splitPoint)] = {"Dy": {}, "Dn": {}}
            # if(len(Dy)>0):
            self.train(X[Dy], Y[Dy], parent[str(splitPoint)]["Dy"])
            # if(len(Dn)>0):
            self.train(X[Dn], Y[Dn], parent[str(splitPoint)]["Dn"])
        else:
            c_ = max(ni_n, key=ni_n.get)
            parent["max"] = c_
            parent["table"] = ni
            return

    def evaluateNumericAttribute(self, X, Y, attrX):
        X = X[:, attrX]
        argS = np.argsort(X)
        X = X[argS]
        Y = Y[argS]
        M = set()
        classes = np.unique(Y)
        ni = {k: 0 for k in classes}
        Nvi = {}
        for j in range(len(X)-1):
            ni[Y[j]] += 1
            if (X[j+1] != X[j]).any():
                v = (X[j+1]+X[j])/2
                M.add(v)
                for i in classes:
                    if not str(v) in Nvi:
                        Nvi[str(v)] = {}
                    Nvi[str(v)][i] = len(
                        [x for x, y in zip(X, Y) if x <= v and y == i])
        ni[Y[-1]] += 1
        _v, bestScore = None, 0
        PciD = {k: v/len(X) for k, v in ni.items()}
        PciDy = {}
        ny = 0
        PciDn = {}
        nn = 0
        for v in M:
            ny = 0
            for i in classes:
                ny += Nvi[str(v)][i]
                PciDy[i] = Nvi[str(v)][i] / sum(Nvi[str(v)].values())
                PciDn[i] = (ni[i]-Nvi[str(v)][i])/sum([ni[j]-Nvi[str(v)][j]
                                                       for j in Nvi[str(v)].keys()])
            nn = len(X)-ny
            score = self.gain.evaluate(
                PciD, len(X), ny, nn, PciDy, PciDn, classes)
            if score > bestScore:
                _v = v
                bestScore = score
        return _v, bestScore

    def evaluateCategoricalAttribute(self, X, Y, attrX):
        X = X[:, attrX]
        classes = np.unique(Y)
        domX = np.unique(X)
        nvi = {}
        ni = {}
        for cla in classes:
            ni[cla] = 0
            nvi[cla] = {}
            for v in domX:
                nvi[cla][v] = 0
        for x, y in zip(X, Y):
            ni[y] += 1
            nvi[y][x] += 1

        PciD = {k: v/len(X) for k, v in ni.items()}
        PciDy = {}
        PciDn = {}
        ny = 0
        nn = 0
        _v, bestScore = None, 0
        for V in list(itertools.combinations(domX, 1)):
            for i in classes:
                ny = sum([1 for v in X if v in V])
                nn = sum([1 for v in X if v not in V])
                try:
                    PciDy[i] = sum([nvi[i][v] for v in domX if v in V]) / sum([sum([nvi[j][v]
                                                                                    for v in domX if v in V]) for j in classes])
                except:
                    PciDy[i] = 0
                try:
                    PciDn[i] = sum([nvi[i][v] for v in domX if v not in V]) / sum([sum([nvi[j][v]
                                                                                        for v in domX if v not in V]) for j in classes])
                except:
                    PciDn[i] = 0
            score = self.gain.evaluate(
                PciD, len(X), ny, nn, PciDy, PciDn, classes)
            if score > bestScore:
                _v = V
                bestScore = score
        return _v, bestScore

    def printTree(self):
        self.formatData(self.tree)

    def formatData(self, t, s=0):
        if not isinstance(t, dict) and not isinstance(t, list):
            print("| "*s+str(t))
        else:
            for key in t:
                print("| "*s+str(key))
                if not isinstance(t, list):
                    self.formatData(t[key], s+1)

    def predict(self, X):
        res = []
        for x in X:
            res.append(self.recursivePredict(self.tree, x))
        return res

    def recursivePredict(self, t, x):
        for key in t:
            if not isinstance(t, list):
                if(key == "max"):
                    return t[key]
                condition = eval(key)
                attr = condition[0]
                val = condition[1]
                if isinstance(x[attr], Number):
                    if(x[attr] <= val):
                        return self.recursivePredict(t[key]["Dy"], x)
                    else:
                        return self.recursivePredict(t[key]["Dn"], x)
                else:
                    if(x[attr] in val):
                        return self.recursivePredict(t[key]["Dy"], x)
                    else:
                        return self.recursivePredict(t[key]["Dn"], x)

    def plotTree(self, fig=None, ax=None):
        classes = list(np.unique(self.data["Y"]))
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1.*i/len(classes)) for i in range(len(classes))]
        if fig == None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for x, y in zip(self.data["X"], self.data["Y"]):
            ax.scatter(x[0], x[1], color=colors[classes.index(y)])
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        self.getLinesRecursive(self.tree, ax, y_min, y_max, x_min, x_max)

    def getLinesRecursive(self, t, ax, y_min, y_max, x_min, x_max):
        for key in t:
            if not isinstance(t, list):
                if(key == "max"):
                    return
                condition = eval(key)
                attr = condition[0]
                val = condition[1]
                saveX_max = x_max
                savey_max = y_max
                if(attr == 0):
                    ax.plot((val, val), (y_min, y_max), 'k-')
                    x_max = val
                else:
                    ax.plot((x_min, x_max), (val, val), 'k-')
                    y_max = val

                self.getLinesRecursive(
                    t[key]["Dy"], ax, y_min, y_max, x_min, x_max)
                x_max = saveX_max
                y_max = savey_max
                if(attr == 0):
                    x_min = val
                else:
                    y_min = val
                self.getLinesRecursive(
                    t[key]["Dn"], ax, y_min, y_max, x_min, x_max)


class informationGain:
    def evaluate(self, PciD, n, ny, nn, PciDy, PciDn, classes):
        return (-sum([PciD[i]*math.log(0.00000001 if PciD[i] == 0 else PciD[i], 2) for i in classes])) - ((ny/(n))*(-sum([PciDy[i]*math.log(0.00000001 if PciDy[i] == 0 else PciDy[i], 2) for i in classes]))+(nn/(n))*(-sum([PciDn[i]*math.log(0.00000001 if PciDn[i] == 0 else PciDn[i], 2) for i in classes])))


class giniIndex:
    def evaluate(self, PciD, n, ny, nn, PciDy, PciDn, classes):
        return (1-sum([PciD[i]**2 for i in classes]))-((ny/(n))*(1-sum([PciDy[i]**2 for i in classes]))+(nn/(n))*(1-sum([PciDn[i]**2 for i in classes])))


class cart:
    def evaluate(self, PciD, n, ny, nn, PciDy, PciDn, classes):
        return (2*(ny/(n))*(nn/(n)) * sum([abs(PciDy[i]-PciDn[i]) for i in classes]))


class Logistic:
    def __init__(self, _step_size=0.01, _eps=0.01, _max_iter=1000):
        self._step_size = _step_size
        self._eps = _eps
        self._max_iter = _max_iter

    def getName(self):
        return "logistic"

    def train(self, _X, _Y):
        _Y = _Y.astype('object')
        d = len(_X[0])
        n = len(_X)
        self.classes = np.unique(_Y)
        dClasses = len(self.classes)
        _X = np.insert(_X, 0, [1 for i in range(n)], axis=1)
        for i in range(n):
            index = np.where(self.classes == _Y[i])[0][0]
            _Y[i] = np.zeros(dClasses)
            _Y[i][index] = 1
            _Y[i] = _Y[i]
        t = 0
        w = {}
        for j in range(dClasses):
            if j not in w:
                w[j] = []
            w[j].append(np.array([0 for i in range(d+1)]))
        temp = np.arange(n)
        np.random.shuffle(temp)
        _X = _X[temp]
        _Y = _Y[temp]
        pi = {}
        gradient = {}
        while True:
            wCopy = {}
            for j in range(dClasses-1):
                wCopy[j] = w[j][-1]
            for x, y in zip(_X, _Y):
                for j in range(dClasses-1):
                    pi[j] = np.exp(
                        np.dot(wCopy[j], x))/sum([np.exp(np.dot(w[a][-1], x)) for a in range(dClasses)])
                    gradient[j] = (y[j]-pi[j])*x
                    wCopy[j] = wCopy[j]+self._step_size*gradient[j]
            for j in range(dClasses-1):
                w[j].append(wCopy[j])
            t += 1
            if sum([np.linalg.norm(w[j][t]-w[j][t-1]) for j in range(dClasses-1)]) <= self._eps or t > self._max_iter:
                break
        for j in range(dClasses):
            w[j] = w[j][-1]
        self.w = w

    def predict(self, _X):
        res = []
        for x in _X:
            temp = {}
            x = np.insert(x, 0, 1, axis=0)
            for c in self.w:
                temp[c] = math.exp(np.dot(
                    self.w[c], x))/sum([math.exp(np.dot(self.w[a], x)) for a in range(len(self.w.keys()))])
            res.append(self.classes[max(temp, key=temp.get)])
        return res

    def predictAttr(self, _X, attr):
        res = []
        for x in _X:
            x = np.insert(x, 0, 1, axis=0)
            res.append(math.exp(np.dot(self.w[attr], x))/sum(
                [math.exp(np.dot(self.w[a], x)) for a in range(len(self.w.keys()))]))
        return res

    def getProbabilites(self, _X):
        matrix = np.zeros((len(_X), len(self.w.keys())))
        for i in range(len(_X)):
            for j in self.w.keys():
                matrix[i][j] = self.predictAttr([_X[i]], j)[0]
        return matrix


def k_cross_validate(learner, X, y, k, repeats):
    scores_in = []
    scores = []
    n = len(X)
    for r in range(repeats):
        arr = list(range(n))
        random.shuffle(arr)
        ks= np.array_split(np.array(arr),k)
        for kIndex in ks:
            indices_test = kIndex
            indices_train = [i for i in arr if not i in indices_test]
            learner.train(X[indices_train,:], y[indices_train])
            
            # compute in-sample error
            y_hat = learner.predict(X[indices_train])
            mistakes = 0
            for i, pred in enumerate(y_hat):
                act = y[indices_train[i]]
                if pred != act:
                    mistakes += 1
            scores_in.append(mistakes / len(indices_train))
            
            # compute validation error
            y_hat = learner.predict(X[indices_test])
            mistakes = 0
            for i, pred in enumerate(y_hat):
                act = y[indices_test[i]]
                if pred != act:
                    mistakes += 1
            scores.append(mistakes / len(indices_test))
            
        return np.mean(scores_in), np.mean(scores)