from pipeline.asp_runner import run_asp
from pipeline.converter import person_to_asp
from interface import predict, asp_to_person

def find_counterfactual(person):

    original_pred = predict(person)

    facts = person_to_asp(person)
    candidates = run_asp(facts)

    for c in candidates:
        new_person = asp_to_person(c)
        pred = predict(new_person)

        if pred != original_pred:
            return new_person

    return None