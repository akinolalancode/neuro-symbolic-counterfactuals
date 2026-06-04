import torch
import time

from interface import predict
from pipeline.run_pipeline import find_counterfactual
from baseline.random_baseline import random_counterfactual
from ml.preprocessing import EDUCATION_DECODE, OCCUPATION_DECODE, SEX_DECODE

# How many test people to evaluate
N = 50


def tensor_row_to_person(row):
    # Reverse the normalisation done in preprocessing.py to get a readable person dict
    age        = round(row[0].item() * 100)
    education  = EDUCATION_DECODE[round(row[1].item() * 2)]
    hours      = round(row[2].item() * 100)
    occupation = OCCUPATION_DECODE[round(row[3].item() * 3)]
    sex        = SEX_DECODE[round(row[4].item())]
    return {"age": age, "education": education, "hours": hours,
            "occupation": occupation, "sex": sex}


def count_changes(original, counterfactual):
    return sum(1 for k in original if original[k] != counterfactual[k])


def is_plausible(person):
    # Check that the counterfactual satisfies the same validity rules ASP enforces
    if person is None:
        return False
    if not (1 <= person["hours"] <= 80):
        return False
    if person["education"] not in ["highschool", "bachelors", "masters"]:
        return False
    if person["occupation"] not in ["tech", "sales", "admin", "bluecollar"]:
        return False
    return True


def run_evaluation():
    data = torch.load("ml/test_data.pt", weights_only=True)
    X = data["X"]

    # Take the first N rows as our test people
    test_people = [tensor_row_to_person(X[i]) for i in range(N)]

    print(f"Evaluating on {N} test examples...")
    print()

    asp_found       = 0
    asp_changes     = []
    asp_plausible   = 0
    asp_times       = []

    base_found      = 0
    base_changes    = []
    base_plausible  = 0
    base_times      = []

    for i, person in enumerate(test_people):
        print(f"  Example {i+1:>2}/{N}  original pred: {predict(person)}", end="  ")

        # --- ASP ---
        t0 = time.time()
        asp_cf = find_counterfactual(person)
        asp_times.append(time.time() - t0)

        if asp_cf is not None:
            asp_found += 1
            asp_changes.append(count_changes(person, asp_cf))
            if is_plausible(asp_cf):
                asp_plausible += 1
            print(f"ASP: found ({count_changes(person, asp_cf)} change/s)", end="  ")
        else:
            print("ASP: not found              ", end="  ")

        # --- Random baseline ---
        t0 = time.time()
        base_cf = random_counterfactual(person)
        base_times.append(time.time() - t0)

        if base_cf is not None:
            base_found += 1
            base_changes.append(count_changes(person, base_cf))
            if is_plausible(base_cf):
                base_plausible += 1
            print(f"Random: found ({count_changes(person, base_cf)} change/s)")
        else:
            print("Random: not found")

    # --- Summary ---
    print()
    print("=" * 55)
    print(f"{'Metric':<30} {'ASP':>10} {'Random':>10}")
    print("=" * 55)

    asp_validity   = asp_found / N * 100
    base_validity  = base_found / N * 100
    print(f"{'Validity (%)':<30} {asp_validity:>9.1f}% {base_validity:>9.1f}%")

    asp_min  = (sum(asp_changes)  / len(asp_changes))  if asp_changes  else 0
    base_min = (sum(base_changes) / len(base_changes)) if base_changes else 0
    print(f"{'Avg features changed':<30} {asp_min:>10.2f} {base_min:>10.2f}")

    asp_plaus  = asp_plausible  / asp_found  * 100 if asp_found  else 0
    base_plaus = base_plausible / base_found * 100 if base_found else 0
    print(f"{'Plausibility (%)':<30} {asp_plaus:>9.1f}% {base_plaus:>9.1f}%")

    asp_rt  = sum(asp_times)  / N
    base_rt = sum(base_times) / N
    print(f"{'Avg runtime (seconds)':<30} {asp_rt:>10.3f} {base_rt:>10.3f}")

    print("=" * 55)
    print()
    print("A few example counterfactuals:")
    print()

    shown = 0
    for i, person in enumerate(test_people):
        if shown >= 3:
            break
        cf = find_counterfactual(person)
        if cf is None:
            continue
        changed_features = [k for k in person if person[k] != cf[k]]
        feature = changed_features[0] if len(changed_features) == 1 else str(changed_features)
        print(f"  Person {i+1}: pred={predict(person)}  ->  counterfactual pred={predict(cf)}")
        print(f"    Original:      {person}")
        print(f"    Counterfactual:{cf}")
        print(f"    Changed:       {changed_features}")
        print()
        shown += 1


if __name__ == "__main__":
    run_evaluation()
