# %%
from sklearn.datasets import load_boston
boston = load_boston()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

rmse = 0

# %%
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [1000, 1500, 2000],
    'gamma': [0.1, 0.4, 0.8],
    'learning_rate': [0.01],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1]
}

model = XGBRegressor(n_estimators=1000, gamma=0.1,
                     learning_rate=0.01, max_depth=5, subsample=0.8)
# model = GridSearchCV(XGBRegressor(), params)
model.fit(X_train, y_train, early_stopping_rounds=10,
          eval_set=[(X_test, y_test)], verbose=True)

# print(model.best_params_)

# %%
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np
print('RMSE Before: ', rmse)
rmse = np.sqrt(mean_squared_error(predictions, y_test))
print('RMSE Current: ', rmse)

import matplotlib.pyplot as plt
plt.plot(y_test, label='test_y')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
