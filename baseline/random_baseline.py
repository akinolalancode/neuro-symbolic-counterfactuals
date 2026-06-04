import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interface import predict
from config import HOURS_DELTAS, EDUCATION_LEVELS, EDITABLE_FEATURES

OCCUPATION_VALUES = ["tech", "sales", "admin", "bluecollar"]


def random_counterfactual(person, max_tries=50):
    original_pred = predict(person)

    for _ in range(max_tries):
        new_person = person.copy()

        # Randomly pick one editable feature and change it
        feature = random.choice(EDITABLE_FEATURES)

        if feature == "hours":
            delta = random.choice([d for d in HOURS_DELTAS if d != 0])
            new_val = new_person["hours"] + delta
            # Keep hours within valid range
            if 1 <= new_val <= 80:
                new_person["hours"] = new_val

        elif feature == "education":
            options = [e for e in EDUCATION_LEVELS if e != new_person["education"]]
            new_person["education"] = random.choice(options)

        elif feature == "occupation":
            options = [o for o in OCCUPATION_VALUES if o != new_person["occupation"]]
            new_person["occupation"] = random.choice(options)

        if predict(new_person) != original_pred:
            return new_person

    return None  # no flip found within max_tries