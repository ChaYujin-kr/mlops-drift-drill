import pickle, bentoml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import split

ref, _, _ = split()
X = pd.get_dummies(ref.drop(columns="class"))
y = (ref["class"] == ">50K").astype(int)

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X, y)
pickle.dump(list(X.columns), open("column_list.pkl", "wb"))
saved = bentoml.sklearn.save_model("income_drill", model, signatures={
    "predict": {"batchable": False}, "predict_proba": {"batchable": False}})
print("saved", saved.tag)
print(f"raw cols={ref.shape[1] - 1}, one-hot cols={X.shape[1]}, pos rate={y.mean():.2f}")
