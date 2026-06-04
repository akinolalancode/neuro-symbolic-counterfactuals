import pandas as pd
import torch

# --- Vocabulary maps: raw CSV strings -> simplified labels used in ASP ---
EDUCATION_MAP = {
    "HS-grad":   "highschool",
    "Bachelors": "bachelors",
    "Masters":   "masters",
}

OCCUPATION_MAP = {
    "Tech-support":      "tech",
    "Exec-managerial":   "tech",
    "Prof-specialty":    "tech",
    "Sales":             "sales",
    "Adm-clerical":      "admin",
    "Protective-serv":   "admin",
    "Priv-house-serv":   "admin",
    "Armed-Forces":      "admin",
    "Craft-repair":      "bluecollar",
    "Machine-op-inspct": "bluecollar",
    "Transport-moving":  "bluecollar",
    "Handlers-cleaners": "bluecollar",
    "Farming-fishing":   "bluecollar",
    "Other-service":     "bluecollar",
}

SEX_MAP = {"Male": "male", "Female": "female"}

# --- Numeric encoding: simplified label -> integer the MLP can use ---
EDUCATION_ENCODE  = {"highschool": 0, "bachelors": 1, "masters": 2}
OCCUPATION_ENCODE = {"tech": 0, "sales": 1, "admin": 2, "bluecollar": 3}
SEX_ENCODE        = {"male": 0, "female": 1}

# Reverse maps: integer -> label (used when reading ASP output back)
EDUCATION_DECODE  = {v: k for k, v in EDUCATION_ENCODE.items()}
OCCUPATION_DECODE = {v: k for k, v in OCCUPATION_ENCODE.items()}
SEX_DECODE        = {v: k for k, v in SEX_ENCODE.items()}

INPUT_SIZE = 5  # age, education, hours, occupation, sex

# Normalisation: divide continuous features so all inputs sit in [0, 1]
# Without this, age (up to 90) dominates education (0-2) in the MLP
AGE_NORM        = 100.0
HOURS_NORM      = 100.0
EDUCATION_NORM  = 2.0   # max encoded value is 2 (masters)
OCCUPATION_NORM = 3.0   # max encoded value is 3 (bluecollar)


def load_data(path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(path, header=None, names=columns, skipinitialspace=True)
    df = df[["age", "education", "hours-per-week", "occupation", "sex", "income"]]
    df = df.rename(columns={"hours-per-week": "hours"})

    # Map raw CSV strings to simplified labels; rows with unknown values become NaN and are dropped
    df["education"]  = df["education"].map(EDUCATION_MAP)
    df["occupation"] = df["occupation"].map(OCCUPATION_MAP)
    df["sex"]        = df["sex"].map(SEX_MAP)
    df = df.dropna()

    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    return df.reset_index(drop=True)


def encode_df(df):
    # Turn a full DataFrame into tensors for training
    X = pd.DataFrame({
        "age":        df["age"].astype(float)                               / AGE_NORM,
        "education":  df["education"].map(EDUCATION_ENCODE).astype(float)   / EDUCATION_NORM,
        "hours":      df["hours"].astype(float)                             / HOURS_NORM,
        "occupation": df["occupation"].map(OCCUPATION_ENCODE).astype(float) / OCCUPATION_NORM,
        "sex":        df["sex"].map(SEX_ENCODE).astype(float),
    })
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(df["income"].values, dtype=torch.float32)
    return X_tensor, y_tensor


def encode_person(person):
    # Turn a single person dict into a 1-row tensor for prediction
    row = [
        float(person["age"])                               / AGE_NORM,
        float(EDUCATION_ENCODE[person["education"]])        / EDUCATION_NORM,
        float(person["hours"])                             / HOURS_NORM,
        float(OCCUPATION_ENCODE[person["occupation"]])     / OCCUPATION_NORM,
        float(SEX_ENCODE[person["sex"]]),
    ]
    return torch.tensor([row], dtype=torch.float32)