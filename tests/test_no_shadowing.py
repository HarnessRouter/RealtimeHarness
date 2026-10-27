"""No class defines a method twice. Python takes the second silently, and the first one's callers then call
the wrong method: a call runner named like the text-turn runner broke every text turn once."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "realtime_harness"


def test_no_method_is_defined_twice_in_a_class():
    twice = []
    for path in sorted(PKG.glob("*.py")):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            seen = set()
            for node in cls.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = {ast.unparse(d) for d in node.decorator_list}
                if any(d.endswith((".setter", ".deleter")) or d.endswith("overload") for d in decorators):
                    continue
                if node.name in seen:
                    twice.append(f"{path.name}: {cls.name}.{node.name}")
                seen.add(node.name)
    assert not twice, twice
