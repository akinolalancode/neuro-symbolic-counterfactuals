# Neuro-Symbolic Counterfactual Explanations

A system that explains ML predictions by finding the smallest valid change
to a person's features that flips the model's income prediction.

The system combines a trained MLP (neural network) with Answer Set Programming
(ASP) using the clingo solver. The MLP predicts income class; ASP generates
all valid candidate modifications; Python finds the one that flips the prediction
with the fewest changes.

## Setup

```
pip install -r requirements.txt
pip install clingo
```

## How to run

**1. Train the model** (only needed once — a trained model.pth is already included):
```
python ml/train.py
```

**2. Run the evaluation** (tests 50 examples, compares ASP vs random baseline):
```
python evaluate.py
```

**3. Try one person manually:**
```python
from pipeline.run_pipeline import find_counterfactual
from interface import predict

person = {"age": 39, "education": "bachelors", "hours": 45, "occupation": "tech", "sex": "male"}
result = find_counterfactual(person)
print("Original prediction:", predict(person))
print("Counterfactual:", result, "-> prediction:", predict(result))
```

## Evaluation results (50 test examples)

| Metric                  | ASP system | Random baseline |
|-------------------------|-----------|----------------|
| Validity (found CF)     | 64.0%     | 32.0%          |
| Avg features changed    | 1.56      | 1.00           |
| Plausibility            | 100.0%    | 100.0%         |
| Avg runtime per example | 0.004s    | 0.001s         |

## Model

MLP: 5 inputs → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid  
Loss: BCELoss | Optimiser: Adam (lr=0.001) | Batch: 256 | Epochs: 100  
Test accuracy: **79.1%** (1,995 / 2,523 correct)

## Project structure

```
neuro-symbolic-counterfactuals/
│
├── config.py           ← defines all feature names and their allowed values
├── interface.py        ← shared helpers: predict(), encode(), asp_to_person()
├── evaluate.py         ← runs the full evaluation on 50 test people and prints results
├── requirements.txt    ← Python dependencies
│
├── data/
│   └── adult.csv       ← UCI Adult Income dataset (48,000 rows, 1994 US Census)
│
├── ml/
│   ├── model.py        ← MLP class definition (5 inputs → 32 → 16 → 1 → Sigmoid)
│   ├── preprocessing.py← loads adult.csv, cleans and normalises features
│   ├── train.py        ← trains the MLP and saves model.pth
│   ├── ml.ipynb        ← same as train.py but as a notebook with visible outputs
│   ├── model.pth       ← saved trained model weights (ready to use)
│   └── test_data.pt    ← saved test split used by evaluate.py
│
├── asp/
│   ├── counterfactual.lp      ← all ASP rules: feature domains, constraints,
│   │                             adjacency transitions and candidate generation
│   └── test_counterfactual.lp ← unit tests that verify the ASP rules are correct
│
├── pipeline/
│   ├── run_pipeline.py ← find_counterfactual(): main entry point that ties
│   │                     ML + ASP together and returns the best counterfactual
│   ├── asp_runner.py   ← calls clingo with the rules + facts, collects all
│   │                     answer sets and sorts them by number of changes
│   └── converter.py    ← converts a person dict into ASP old_value/1 facts
│                         so clingo can reason about the current values
│
└── baseline/
    └── random_baseline.py ← random search comparison: randomly mutates features
                              until the prediction flips, used in evaluate.py
```

## Features used

| Feature    | Type        | Fixed / Editable | Allowed values                     |
|------------|-------------|------------------|------------------------------------|
| age        | numeric     | fixed            | unchanged                          |
| sex        | categorical | fixed            | unchanged                          |
| education  | categorical | editable         | highschool, bachelors, masters     |
| hours      | numeric     | editable         | original ± 5 or ± 10 (steps of 5) |
| occupation | categorical | editable         | tech, sales, admin, bluecollar     |
