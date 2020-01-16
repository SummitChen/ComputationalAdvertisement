from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

from hyperopt import hp, fmin, rand, anneal, tpe, space_eval, Trials

import numpy as np
import matplotlib.pyplot as plt

class Optimization:
    def __init__(self, train_x, test_x, train_y, test_y):
        super().__init__()
        self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y
        self.max_depth_range = np.arange(2, 200)
        self.samples_split_range = np.arange(2, 200)
        self.samples_leaf_range = [1, 2]
        self.random_state_range = np.arange(2, 200)

        self.space4cart = {
            'max_depth': hp.choice('max_depth',
                self.max_depth_range),
            'min_samples_split': hp.choice('min_samples_split', self.samples_split_range),
            'min_samples_leaf': hp.choice('min_samples_leaf', self.samples_leaf_range),
            'random_state': hp.choice('random_state', self.random_state_range)
        }
    
    def optimize_func (self, args):
        cart = DecisionTreeClassifier(**args)
        cart.fit(self.train_x, self.train_y)
        predict_y = cart.predict(self.test_x)
        return -accuracy_score(predict_y, self.test_y)

    def search_best_hyperparameters(self, eval_iters):

        best = fmin(self.optimize_func, self.space4cart, algo = tpe.suggest, max_evals = eval_iters)

        best['max_depth'] = self.max_depth_range[best['max_depth']]
        best['min_samples_split'] = self.samples_split_range[best['min_samples_split']]
        best['min_samples_leaf'] = self.samples_leaf_range[best['min_samples_leaf']]
        best['random_state'] = self.random_state_range[best['random_state']]

        return best

class Preprocess:

    @staticmethod
    def load_data():
        digits = load_digits()
        return digits.data, digits

    @staticmethod
    def explore_data(data, digits):
        print(data.shape)
        print(digits.images[0])
        print(digits.target[0])

        plt.gray()
        plt.title('Handwritten Digits')
        plt.imshow(digits.images[0])
        plt.show()

    @staticmethod
    def split_data(data, digits):
        return train_test_split(data, digits.target, test_size = 0.25, random_state=33)

    @staticmethod
    # Z-Score Normalization
    def z_score_normalize(train_x, test_x):
        ss = preprocessing.StandardScaler()
        train_ss_x = ss.fit_transform(train_x)
        test_ss_x = ss.transform(test_x)
        return train_ss_x, test_ss_x

def main():
    data, digits = Preprocess.load_data()
    Preprocess.explore_data(data, digits)
    train_x, test_x, train_y, test_y = Preprocess.split_data(data, digits)
    train_x, test_x = Preprocess.z_score_normalize(train_x, test_x)

    optimizer = Optimization(train_x, test_x, train_y, test_y)
    best = optimizer.search_best_hyperparameters(5000)
    print(best)
    print(DecisionTreeClassifier(**best))
    print('Best CART model accuracy: {0:5.4f}'.format(-optimizer.optimize_func(best)))

    last_best = {
        'max_depth': 65,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'random_state': 187
    }

    print('Last Best CART model accuracy: {0:5.4f}'.format(-optimizer.optimize_func(last_best)))

if __name__ == "__main__":
    main()