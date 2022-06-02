from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from dtreeviz.trees import dtreeviz

import numpy as np

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

cls = tree.DecisionTreeClassifier()  # limit depth of tree

wk = {
    'Sun': 0,
    'Mon': 1,
    'Tue': 1,
    'Wed': 1,
    'Thu': 1,
    'Fri': 1,
    'Sat': 0,
}

ss = {
    'Spring': 0,
    'Summer': 1,
    'Autumn': 2,
    'Winter': 3,
}

wt = {
    'Shiny': True,
    'Rainy': False,
}

rs = {
    'Bad': 0,
    'Good': 1,
}

# 교통 체증 True

data = np.array([
    [ss['Spring'], wk['Sun'], 9, wt['Shiny'], rs['Good']],
    [ss['Spring'], wk['Mon'], 8, wt['Shiny'], rs['Bad']],
    [ss['Summer'], wk['Sun'], 8, wt['Rainy'], rs['Good']],
    [ss['Autumn'], wk['Sun'], 13, wt['Rainy'], rs['Bad']],
    [ss['Autumn'], wk['Tue'], 14, wt['Rainy'], rs['Good']],
    [ss['Autumn'], wk['Mon'], 8, wt['Rainy'], rs['Bad']],
    [ss['Winter'], wk['Sat'], 8, wt['Shiny'], rs['Good']],
    [ss['Winter'], wk['Sun'], 9, wt['Shiny'], rs['Good']],
    [ss['Winter'], wk['Sun'], 10, wt['Shiny'], rs['Good']],
    [ss['Winter'], wk['Mon'], 13, wt['Shiny'], rs['Good']],
])

X = data[:, :4]
y = data[:, 4:].flatten()

cls.fit(X, y)
print(cls.score(X, y))
print(cls.predict([[ss['Winter'], wk['Mon'], 9, wt['Shiny']]]))

cls2 = ensemble.RandomForestClassifier()
cls2.fit(X, y)
print(cls2.predict([[ss['Winter'], wk['Mon'], 9, wt['Shiny']]]))

viz = dtreeviz(cls,
               X,
               y,
               target_name='Traffic Jam',
               feature_names=['Season', 'Weekday', 'Hour', 'Weather'],
               class_names=['Bad', 'Good']  # need class_names for classifier
               )

viz.view()
