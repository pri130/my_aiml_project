import pickle
import pytest
import os

def test_model_loading():
    """Test if the model loads correctly."""
    try:
        # Go up 2 levels from tests/ to reach project root, then into model/
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)),
            "model", "model.pkl")
        )
        print(f"Looking for model at: {model_path}")  # Debug print
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert model is not None
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")