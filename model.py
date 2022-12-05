import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

university_df = pd.read_csv(
    r"C:\Users\RupanshuJanbandhu\Downloads\MLOps\MLOps\16 "
    + r"XGboost in SKLearn\university_admission.csv"
)

X = university_df.drop(columns=["Chance_of_Admission"])
y = university_df["Chance_of_Admission"]

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)

print(tree.predict(X_test))
