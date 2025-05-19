# predict.py
import pickle
import numpy as np

# 1. Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Example prediction (replace with your input data)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example feature vector
prediction = model.predict(new_data)

print(f"Prediction: {prediction}")
