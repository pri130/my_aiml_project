import pickle
import pytest

def test_model_loading():
    """Test if the model loads correctly."""
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None

# Add more tests for your scripts!
