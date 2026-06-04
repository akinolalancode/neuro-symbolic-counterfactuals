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

% Allowed numeric modifications for working hours
delta_hours(-10; -5; 0; 5; 10).

% Allowed education values
education_value(highschool; bachelors; masters).

% Allowed occupation values
occupation_value(tech; sales; admin; bluecollar).

% =========================================================
% 2a. OBSERVED INSTANCE
% =========================================================

old_value(hours,40).

old_value(education,bachelors).

old_value(occupation,tech).

old_value(age,30).

old_value(sex,male).

% =========================================================
% 3. HOURS MODIFICATION
% =========================================================

1 { choose_hours(D) : delta_hours(D) } 1.

1 {
    new_value(hours, V + D) :
        old_value(hours,V),
        choose_hours(D)
} 1.

% =========================================================
% 4. EDUCATION MODIFICATION
% =========================================================

can_change_education(highschool,bachelors).

can_change_education(bachelors,masters).

can_change_education(bachelors,highschool).

can_change_education(masters,bachelors).

% =========================================================
% 5. FIXED FEATURES
% =========================================================

1 { new_value(age,V) : old_value(age,V) } 1.

1 { new_value(sex,V) : old_value(sex,V) } 1.

1 {
    new_value(education,E);
    new_value(education,E2) :
        can_change_education(E,E2)
} 1 :-
    old_value(education,E).

% =========================================================
% 6. OCCUPATION MODIFICATION
% =========================================================

1 {
    new_value(occupation,O);
    new_value(occupation,O2) :
        occupation_value(O2),
        O2 != O
} 1 :-
    old_value(occupation,O).

% =========================================================
% 7. CHANGE DETECTION
% =========================================================

changed(hours) :-
    old_value(hours,V1),
    new_value(hours,V2),
    V1 != V2.

changed(education) :-
    old_value(education,V1),
    new_value(education,V2),
    V1 != V2.

changed(occupation) :-
    old_value(occupation,V1),
    new_value(occupation,V2),
    V1 != V2.

% =========================================================
% 8. VALIDITY CONSTRAINTS
% =========================================================

:- new_value(hours,V), V < 1.

:- new_value(hours,V), V > 80.

% =========================================================
% 9. SAFETY RULES
% =========================================================

:- new_value(F,V1),
   new_value(F,V2),
   V1 != V2.

% =========================================================
% 10. OPTIMIZATION
% =========================================================

#minimize {
    1@1 : changed(hours);
    1@1 : changed(education);
    1@1 : changed(occupation)
}.

% =========================================================
% 11. OUTPUT
% =========================================================

#show new_value/2.
#show changed/1.