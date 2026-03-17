import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def train_model(data):

  X = data[[
    "MA_ratio", "Price_to_MA50",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Volatility", "Volatility_10",
    "Volume_ratio",
    "RSI", "BB_position"
]]
  y = data["Target"]

  split = int(len(data) * 0.8)
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]

  # Pipeline (No scaler for RandomForest so,)
  pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
          random_state=42,
          objective="multi:softmax",
          num_class=3,
          eval_metric="logloss",
          use_label_encoder=False
        ))
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