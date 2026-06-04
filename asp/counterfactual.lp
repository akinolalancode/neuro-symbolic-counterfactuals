% =========================================================
% COUNTERFACTUAL GENERATION WITH ASP (CLINGO)
% =========================================================

% =========================================================
% 1. EDITABLE AND FIXED FEATURES
% =========================================================

editable(hours).
editable(education).
editable(occupation).

fixed(age).
fixed(sex).

% =========================================================
% 2. DOMAIN DEFINITIONS
% =========================================================

% Allowed deltas for hours (numeric feature)
delta_hours(-10; -5; 0; 5; 10).

% Allowed values for categorical features
education_value(highschool; bachelors; masters).
occupation_value(tech; sales; admin; bluecollar).

% =========================================================
% 3. HOURS MODIFICATION
% =========================================================

% Pick one delta, apply it to the old value
1 { choose_hours(D) : delta_hours(D) } 1.

1 {
    new_value(hours, V + D) :
        old_value(hours, V),
        choose_hours(D)
} 1.

% =========================================================
% 4. FIXED FEATURES
% =========================================================

% Age and sex cannot change -- copy old value straight through
1 { new_value(age, V) : old_value(age, V) } 1.
1 { new_value(sex, V) : old_value(sex, V) } 1.

% =========================================================
% 5. EDUCATION MODIFICATION
% =========================================================

% Only adjacent transitions are allowed (no jumping from highschool to masters)
can_change_education(highschool, bachelors).
can_change_education(bachelors, masters).
can_change_education(bachelors, highschool).
can_change_education(masters, bachelors).

% Either keep the same level or move to an adjacent one
1 {
    new_value(education, E);
    new_value(education, E2) : can_change_education(E, E2)
} 1 :-
    old_value(education, E).

% =========================================================
% 6. OCCUPATION MODIFICATION
% =========================================================

% Can keep the same occupation or switch to any other valid one
1 {
    new_value(occupation, O);
    new_value(occupation, O2) : occupation_value(O2), O2 != O
} 1 :-
    old_value(occupation, O).

% =========================================================
% 7. CHANGE DETECTION
% =========================================================

% A feature is changed if old and new values differ
changed(F) :- old_value(F, V1), new_value(F, V2), V1 != V2.

% =========================================================
% 8. SAFETY
% =========================================================

% Each feature must have exactly one new value
:- new_value(F, V1), new_value(F, V2), V1 != V2.

% =========================================================
% 9. VALIDITY CONSTRAINTS
% =========================================================

% Hours must stay in a realistic range
:- new_value(hours, V), V < 1.
:- new_value(hours, V), V > 80.

% =========================================================
% 10. OUTPUT
% =========================================================

#show new_value/2.
#show changed/1.