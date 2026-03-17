import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score

def train_model(data):

  X = data[["MA50","MA200","Volatility","Volume"]]
  y = data["Target"]

  split = int(len(data) * 0.8)
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]

  # Pipeline (No scaler for RandomForest so,)
  pipeline = Pipeline([
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
  ])

  # Time Series cross-validation
  tscv = TimeSeriesSplit(n_splits=5)

  cv_scores = cross_val_score(
    pipeline, X_train, y_train, 
    cv = tscv,
    scoring = "accuracy"
  )

  # Train Final model
  pipeline.fit(X_train, y_train)

  # Test Prediction
  preds = pipeline.predict(X_test)
  test_accuracy = accuracy_score(y_test, preds)

  return pipeline, test_accuracy, cv_scores.mean()