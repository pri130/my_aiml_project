# train.py
import pandas as pd
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import train_test_split

# 1. Load data
data = pd.read_csv("data/training_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# 4. Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
