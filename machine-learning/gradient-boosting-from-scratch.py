# %%
import pandas as pd
from sklearn.model_selection import train_test_split

df_bikes = pd.read_csv('./machine-learning/data/bike_rentals_cleaned.csv')

X_bikes = df_bikes.iloc[:, :-1]
y_bikes = df_bikes.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_bikes, y_bikes, random_state=2)

# %%
from sklearn.tree import DecisionTreeRegressor

y1_train = y_train

tree_1 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree_1.fit(X_train, y1_train)
y1_train_pred = tree_1.predict(X_train)

y2_train = y1_train - y1_train_pred

tree_2 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree_2.fit(X_train, y2_train)
y2_train_pred = tree_2.predict(X_train)

y3_train = y2_train - y2_train_pred

tree_3 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree_3.fit(X_train, y3_train)
y3_train_pred = tree_3.predict(X_train)

# %%
y1_pred = tree_1.predict(X_test)
y2_pred = tree_2.predict(X_test)
y3_pred = tree_3.predict(X_test)

from sklearn.metrics import mean_squared_error as MSE

print('y1_pred:', MSE(y_test, y1_pred, squared=False))
print('y2_pred:', MSE(y_test, y1_pred + y2_pred, squared=False))
print('y3_pred:', MSE(y_test, y1_pred + y2_pred + y3_pred, squared=False))
