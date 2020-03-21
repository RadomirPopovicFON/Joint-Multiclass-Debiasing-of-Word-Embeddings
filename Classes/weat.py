from __future__ import division
import numpy as np
from random import sample
from itertools import combinations

class WEAT:
    def __init__(self, embedding, X, Y, A, B, steps=-1):
        self.embedding = embedding
        if len(X) != len(Y):
            #print('Warning: len(X) ==', len(X), '!= len(Y) ==', len(Y), ' only considering first', min(len(X), len(Y)), 'elements.')
            pass
        self.X = X[:len(Y)]
        self.Y = Y[:len(X)]
        self.A = A
        self.B = B
        self.steps = steps
        self.a = [embedding.get_value(a) / np.linalg.norm(embedding.get_value(a)) for a in A]
        self.b = [embedding.get_value(b) / np.linalg.norm(embedding.get_value(b)) for b in B]
        self.s = {}
        for w in self.X + self.Y:
            wn = embedding.get_value(w) / np.linalg.norm(embedding.get_value(w))
            self.s[w] = np.mean([np.dot(wn, a) for a in self.a], axis = 0) - np.mean([np.dot(wn, b) for b in self.b], axis = 0)
        self.p_val = self.p()
        self.effect_size_val = self.effect_size()

    def get_p(self):
        return self.p_val

    def get_effect_size(self):
        return self.get_effect_size

    def get_stats(self):
        return self.p_val, self.effect_size_val

    def statistic(self, X, Y):
        return np.sum([self.s[x] for x in X]) - np.sum([self.s[y] for y in Y])

    def p(self):
        base_statistic = self.statistic(self.X, self.Y)
        total = 0
        larger = 1
        if self.steps == -1:
            iterator = combinations(self.X + self.Y, len(self.X + self.Y) // 2)
        else:
            iterator = [sample(self.X + self.Y, len(self.X + self.Y) // 2) for _ in range(self.steps)]
        for Xi in iterator:
            Yi = [y for y in self.X + self.Y if not y in Xi]
            total += 1
            if self.statistic(Xi, Yi) > base_statistic:
                larger += 1
        if(larger == 1): return 0
        if(larger == total+1): return 1
        return larger / total

    def effect_size(self):
        std_words = self.X + self.Y
        std_wns = np.array([self.embedding.get_value(w) for w in std_words])
        std_wns = std_wns / \
                  np.linalg.norm(std_wns, axis=1, keepdims=True)
        std_ss = np.mean([np.matmul(std_wns, a) for a in self.a], axis=0) - np.mean([np.matmul(std_wns, b) for b in self.b], axis=0)
        return (np.mean([self.s[x] for x in self.X]) - np.mean([self.s[y] for y in self.Y])) / np.std(std_ss)