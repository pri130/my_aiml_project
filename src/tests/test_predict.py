import pytest
import os
import xgboost as xgb  # For XGBoost models

def test_model_loading():
    """Test if the XGBoost model loads correctly."""
    try:
        # Get absolute path to model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, "model", "debris_model.json")
        
        print(f"Looking for model at: {model_path}")
        
        # Verify file exists
        if not os.path.exists(model_path):
            pytest.fail(f"Model file not found at: {model_path}")
        
        # Load XGBoost model (native format)
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        assert model is not None
        print("✅ XGBoost model loaded successfully!")
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")