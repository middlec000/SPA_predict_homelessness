import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.uniform(low=0, high=10, size=15).reshape(-1, 1)
y = [1,0,1,1,0,0,0,1,0,1,0,1,0,0,0]

model = LogisticRegression().fit(
    X=X, 
    y=y)

X_test = np.random.uniform(low=0, high=10, size=4).reshape(-1, 1)
predictions = model.predict_proba(
    X=X_test)

print(model.classes_)
print(predictions)
print(predictions[:,1])