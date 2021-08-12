import numpy as np
import matplotlib.pyplot as plt
import random
import itertools as it

def cross_validate(learner, X, y, train_size, repeats):
    scores_in = []
    scores = []
    n = len(X)
    num_examples = int(train_size * n)
    
    for r in range(repeats):
        indices_train = random.sample(range(n), num_examples)
        learner.train(X[indices_train,:], y[indices_train])
        indices_test = [i for i in range(n) if not i in indices_train]
        
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

def plot_learning_curves(learner, X, y, ax = None, train_portions = [0.05, 0.1, 0.2, 0.5, 0.7], repeats=10):
    scores_in = []
    scores_val = []
    for train_portion in train_portions:
        e_in, e_out = cross_validate(learner, X, y, train_portion, repeats)
        scores_in.append(e_in)
        scores_val.append(e_out)
    
    if ax is None:
        fig, ax = plt.subplots()
    x_axis = np.round(len(X) * np.array(train_portions)).astype(int)
    ax.plot(x_axis, scores_in, label="in-sample-error"+learner.getName())
    ax.plot(x_axis, scores_val, label="out-of-sample-error"+learner.getName())
    ax.set_ylim([0,1])
    ax.legend()
    return ax
    

'''
    This computes all instances that are (believed to be) possible in the instance space.
    Given the input matrix, a domain is computed for each attribute (column).
    Then, the cartesian product is built over all the domains.
'''
def get_instance_space(X):
    domains = []
    combos = 1
    for col in X.T:
        domain = list(np.unique(col))
        domains.append(domain)
        combos *= len(domain)
    if combos > 10**6:
        raise Exception("Instance space has " + str(combos) + " instances, which is more than the allowed 10^6")
    return list(it.product(*domains))