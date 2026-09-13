import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "v10" / "train_wavehunter_v10.py"


def test_v10_bayesian_volatility_default_is_disabled():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--bayes-vol-enabled":
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                matches.append(ast.literal_eval(kw.value))
    assert matches == [False], f"unexpected v10 default: {matches}"
