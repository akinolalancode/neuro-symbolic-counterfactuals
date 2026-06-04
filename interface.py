import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.model import MLP
from ml.preprocessing import encode_person, INPUT_SIZE

# Load model once when this file is imported, so every call to predict()
# reuses the same weights instead of reloading from disk each time
_ROOT      = os.path.dirname(os.path.abspath(__file__))
_MODEL_PTH = os.path.join(_ROOT, "ml", "model.pth")

_model = MLP(INPUT_SIZE)
_model.load_state_dict(torch.load(_MODEL_PTH, weights_only=True))
_model.eval()


def encode(person: dict):
    # Converts a person dict to a 1-row float tensor the MLP can accept
    return encode_person(person)


def predict(person: dict) -> int:
    # Returns 1 (income > 50K) or 0 (income <= 50K)
    with torch.no_grad():
        prob = _model(encode(person)).squeeze().item()
    return 1 if prob >= 0.5 else 0


def asp_to_person(symbols) -> dict:
    # Converts clingo answer set symbols back into a person dict
    # e.g. new_value(hours, 40) -> {"hours": 40, ...}
    person = {}
    numeric = {"age", "hours"}
    for s in symbols:
        if s.name == "new_value" and len(s.arguments) == 2:
            feature = str(s.arguments[0])
            value   = str(s.arguments[1])
            person[feature] = int(value) if feature in numeric else value
    return person