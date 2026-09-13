import ast
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "v10" / "train_wavehunter_v10.py"


def _default(flag):
    tree=ast.parse(SCRIPT.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==flag:
            return next(ast.literal_eval(k.value) for k in n.keywords if k.arg=='default')
    raise AssertionError(flag)


def test_hhi_default_is_disabled():
    assert _default('--hhi-enabled') is False


def test_hhi_default_scale_is_mild():
    assert _default('--hhi-weight') == 0.5
    assert _default('--hhi-max-penalty') == 0.5
