
from ipidefense.runner import Run, Step, main, root_arg

TASKACC = "ipidefense.experiments.taskacc"
JUDGE = "ipidefense.experiments.judge"
JUDGE_VALIDATE = "ipidefense.experiments.judge_validate"
IFEVAL_LOGGED = "ipidefense.evaluation.ifeval_logged"
LEGIT = "ipidefense.experiments.legit"
NONWITNESS = "ipidefense.experiments.nonwitness"

RUNS = root_arg("IPI_RUNS", "runs")


def taskacc(run: Run) -> None:
    # items are shipped in data/taskacc, prepare only fills gaps
    run.py(TASKACC, "prepare")
    run.py(TASKACC, "run", "--spot-unbatched", "50")
    run.py(TASKACC, "summarize")


def squad(run: Run) -> None:
    run.py("ipidefense.experiments.squad_rescore")


def judge(run: Run) -> None:
    # the judge labels come from the shipped passes in data/judge
    run.py("ipidefense.experiments.judge_dump", "run", "--seeds", "0", "1", "2")
    run.py(JUDGE, "split", "--dump", f"{RUNS}/judge_dump", "--pass", "A")
    run.py(JUDGE, "split", "--dump", f"{RUNS}/judge_dump", "--pass", "B")
    run.py(JUDGE, "score")
    run.py(JUDGE_VALIDATE, "sample")
    run.py(JUDGE_VALIDATE, "report")


def ifeval(run: Run) -> None:
    # the hardened adapters are batched (runs/ifeval_logged_b16)
    run.py("ipidefense.experiments.ifeval_subsets")
    run.py(IFEVAL_LOGGED, "--arms", "undef", "def-s0", "def-s1", "def-s2")
    run.py("ipidefense.experiments.ifeval_breakdown")
    run.py(IFEVAL_LOGGED, "--batch", "16", "--arms", "hard-s0", "hard-s1")


def legit(run: Run) -> None:
    # documents are shipped, documents.py checks them
    run.py("ipidefense.data.documents")
    run.py(LEGIT, "run", "--exp", "directives", "--seeds", "0", "1", "2", "--unbatched", "300",
           "--batch", "50")
    run.py(LEGIT, "run", "--exp", "docs", "--unbatched", "240", "--batch", "48")


def nonwitness(run: Run) -> None:
    # every language must render from the reviewed wrappers
    run.py(NONWITNESS, "check")
    run.py(NONWITNESS, "run")
    run.py(NONWITNESS, "summarize")


STEPS = [
    Step("taskacc", "host-task accuracy under attack (SQuAD, BoolQ, HotpotQA)",
         "Task accuracy under attack", taskacc),
    Step("squad", "official SQuAD EM / F1 of the stored answer-substitution outputs",
         "SQuAD EM / F1 of the answer-substitution runs", squad),
    Step("judge", "response pairs for the LLM judge; labels and agreement of the shipped passes",
         "LLM-judge table and judge agreement", judge),
    Step("ifeval", "IFEval subset labels, per-prompt logs and breakdown; hardened adapters",
         "IFEval breakdown; IFEval of the hardened adapters", ifeval),
    Step("legit", "legitimate instructions in the data channel (directives, documents)",
         "Legitimate instructions", legit),
    Step("nonwitness", "non-witness objectives in eight languages: rendering check, generation, "
         "summary", "Non-witness objectives in eight languages", nonwitness),
]


if __name__ == "__main__":
    main(__file__, STEPS)
