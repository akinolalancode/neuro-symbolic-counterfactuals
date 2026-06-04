def person_to_asp(person):
    return f"""
    old_value(age,{person['age']}).
    old_value(sex,{person['sex']}).
    old_value(education,{person['education']}).
    old_value(hours,{person['hours']}).
    old_value(occupation,{person['occupation']}).
    """
