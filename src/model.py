import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import pandas as pd

def train_model(data):

  X = data[[
    "MA_ratio", "Price_to_MA50",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Volatility", "Volatility_10",
    "Volume_ratio",
    "RSI", "BB_position"
  ]]
  y = data["Target"]
  X, y = X.align(y, join="inner", axis=0)

  split = int(len(data) * 0.8)
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]

  # Pipeline (No scaler for RandomForest so,)
  pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
          random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,       
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,       
        gamma=0.1,               
        reg_alpha=0.1,            
        reg_lambda=1.5
        ))
    ])

  # Time Series cross-validation
  n_splits = min(5, len(X_train) // 50)  # at least 50 samples per fold
  n_splits = max(2, n_splits)            # minimum 2 splits
  tscv = TimeSeriesSplit(n_splits=n_splits)

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


def save_model(model, accuracy, cv_score, path="model.pkl"):
    joblib.dump({
        "model": model,
        "accuracy": accuracy,
        "cv_score": cv_score
    }, path)

def load_model(path="model.pkl"):
    data = joblib.load(path)
    return data["model"], data["accuracy"], data["cv_score"]




if __name__ == "__main__":

    from data_loader import load_stock_data
    from features import add_features

    print("Training model...")

    tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

    all_data = []

    for t in tickers:
        try:
            data = load_stock_data(t)
            data = add_features(data)
            all_data.append(data)
            print(f"{t} loaded")
        except Exception as e:
            print(f"Error with {t}: {e}")

    final_data = pd.concat(all_data).reset_index(drop=True)
    final_data = final_data.dropna()

    model, acc, cv = train_model(final_data)

    save_model(model, acc, cv)

    print("Model saved as model.pkl")
    print(f"Accuracy: {round(acc*100,2)}%")
    print(f"CV Score: {round(cv*100,2)}%")