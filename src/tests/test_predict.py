import pickle
import pytest
import os

def test_model_loading():
    """Test if the model loads correctly."""
    try:
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, "model", "model.pkl")
        
        print(f"Looking for model at: {model_path}")  # Debug print
        
        # Verify the file exists before trying to open it
        if not os.path.exists(model_path):
            pytest.fail(f"Model file not found at: {model_path}")
            
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        assert model is not None
        print("Model loaded successfully!")
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")