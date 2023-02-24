import joblib

clf = joblib.load("model/bayesian.pkl")

print(clf)