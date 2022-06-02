from sklearn import datasets, tree, model_selection, metrics
import numpy as np


class GradientBooster:
    def __init__(self, n_trees=20):
        self.f = []
        self.learning_rates = []
        self.n_trees = n_trees

    def fit(self, x, y, lr=0.1):
        class F0:
            predict = lambda x: np.mean(y) * np.ones(x.shape[0])

        self.f.append(F0)
        self.learning_rates.append(1)

        for _ in range(self.n_trees):
            m = tree.DecisionTreeRegressor(max_depth=5)
            res = y - self.predict(x)
            m.fit(x, res)
            self.f.append(m)
            self.learning_rates.append(lr)

    def predict(self, x):
        s = 0
        for f, lr in zip(self.f, self.learning_rates):
            # print(f.predict(x) * lr)
            s += f.predict(x) * lr
        # print(s)
        return s


# Some data
np.random.seed(123)
x = datasets.load_diabetes()['data']
y = datasets.load_diabetes()['target']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)


def evaluate(m):
    print('Training score:', metrics.r2_score(y_train, m.predict(x_train)),
          '\tTesting score:', metrics.r2_score(y_test, m.predict(x_test)))


if __name__ == '__main__':
    # Algorithm to beat
    p = {'max_depth': [5, 10, 15, 20],
         'min_samples_split': [2, 3, 7],
         'min_samples_leaf': [1, 3, 7]}

    m1 = model_selection.GridSearchCV(tree.DecisionTreeRegressor(), p)
    # m = tree.DecisionTreeRegressor()
    m1.fit(x_train, y_train)
    evaluate(m1)

    m2 = GradientBooster(20)
    m2.fit(x_train, y_train)
    evaluate(m2)
