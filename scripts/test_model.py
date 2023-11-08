import pandas as pd
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

test = pd.read_csv('/home/petr/project/datasets/data_test.csv')

y_test = test['y'].astype('int').values
X_test = test.drop('y', axis=1).values

model = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=100),
                                  n_shapelet_samples=100,
                                  max_shapelets=100,
                                  batch_size=20)
with open('/home/petr/project/models/data.pickle', 'rb') as f:
    model = pickle.load(f)

test_score = model.score(X_test, y_test)

print(test_score)
