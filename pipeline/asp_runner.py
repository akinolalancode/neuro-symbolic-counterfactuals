import clingo
import os

def run_asp(facts: str):
    # Build an absolute path so this works no matter where the script is run from
    lp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "asp", "counterfactual.lp")

    # --opt-mode=enum enumerates ALL valid answer sets, not just the optimal ones.
    # We need this because ASP doesn't know which candidates will flip the ML prediction --
    # Python has to check each one. We sort by number of changes ourselves so we still
    # try the most minimal candidates first, which is what #minimize in the ASP file expresses.
    ctl = clingo.Control(["--opt-mode=enum", "--models=0"])
    ctl.load(lp_path)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    models = []
    with ctl.solve(yield_=True) as handle:
        for m in handle:
            syms = list(m.symbols(shown=True))
            n_changed = sum(1 for s in syms if s.name == "changed")
            models.append((n_changed, syms))

    # Sort so candidates with fewer changes come first
    models.sort(key=lambda x: x[0])
    return [syms for _, syms in models]