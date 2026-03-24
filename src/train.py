from model import train_model
from data_loader import load_stock_data
import joblib

# Load data
data = load_stock_data()

# Train model
model, accuracy, cv_scores = train_model(data)

# Save model
joblib.dump({
    "model": model,
    "accuracy": accuracy,
    "cv_score": cv_scores
}, "model.pkl")

print("Model saved successfully ✅")