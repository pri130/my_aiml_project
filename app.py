from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import numpy as np
from tle_processor import process_tle

# Initialize Flask
app = Flask(__name__)

# Load model and threshold
model = XGBClassifier()
model.load_model('model/debris_model.json')
threshold = np.load('model/threshold.npy')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for collision predictions"""
    try:
        data = request.json
        features = process_tle(data['tle'])
        proba = model.predict_proba(features)[0, 1]
        
        return jsonify({
            "risk": bool(proba > threshold),
            "probability": float(proba),
            "threshold": float(threshold),
            "features": features.tolist()[0]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "threshold": float(threshold)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
