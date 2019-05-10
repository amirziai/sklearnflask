# Flask API for scikit learn
A simple Flask application that can serve predictions from a scikit-learn model. Reads a pickled sklearn model into memory when the Flask app is started and returns predictions through the /predict endpoint. You can also use the /train endpoint to train/retrain the model. Any sklearn model can be used for prediction.

Read more in [this blog post](https://medium.com/@amirziai/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa).

### Dependencies
- scikit-learn
- Flask
- pandas
- numpy

```
pip install -r requirements.txt
```

### Running API
```
python main.py <port>
```

# Endpoints
### /predict (POST)
Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
[
    {"Age": 85, "Sex": "male", "Embarked": "S"},
    {"Age": 24, "Sex": "female", "Embarked": "C"},
    {"Age": 3, "Sex": "male", "Embarked": "C"},
    {"Age": 21, "Sex": "male", "Embarked": "S"}
]
```

and sample output:
```
{"prediction": [0, 1, 1, 0]}
```


### /train (GET)
Trains the model. This is currently hard-coded to be a random forest model that is run on a subset of columns of the titanic dataset.

### /wipe (GET)
Removes the trained model.
