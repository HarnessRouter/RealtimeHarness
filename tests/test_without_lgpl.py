"""The harness runs in an install without Pipecat's two LGPL packages, `soxr` and `num2words`.

Each check runs in a process of its own where both packages cannot be imported, whether or not this
environment has them. CI also runs the whole suite with them uninstalled (.github/workflows/test.yml).
"""
import ast
import pathlib
import subprocess
import sys
import textwrap

PKG = pathlib.Path(__file__).resolve().parent.parent / "realtime_harness"

# a module that is None in sys.modules cannot be imported, and importlib reports it as not found
WITHOUT = "import sys\nsys.modules['soxr'] = None\nsys.modules['num2words'] = None\n"


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", WITHOUT + textwrap.dedent(body)], capture_output=True, text=True,
                          cwd=PKG.parent, timeout=120)


def pipecat_imports() -> list[str]:
    """Every `from pipecat… import …` the harness has, as written."""
    found = set()
    for path in sorted(PKG.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pipecat":
                found.add(ast.unparse(node))
            if isinstance(node, ast.Import):
                found.update(f"import {a.name}" for a in node.names if a.name.split(".")[0] == "pipecat")
    return sorted(found)


def test_the_harness_imports_pipecat_somewhere():
    # the check below is only worth something while this list is not empty
    assert any("realtime" in line for line in pipecat_imports())


def test_everything_the_harness_imports_from_pipecat_loads_without_the_lgpl_packages():
    lines = "\n".join(line for line in pipecat_imports() if "smallwebrtc" not in line)
    r = _run("import realtime_harness\n" + lines + "\n"
             "import soxr\nassert soxr.STAND_IN\n"
             "assert sys.modules['num2words'] is None\nprint('loaded')\n")
    assert r.returncode == 0 and r.stdout.strip().endswith("loaded"), r.stderr[-2000:]


def test_without_the_stand_in_the_same_imports_fail():
    # what the stand-in is for: were Pipecat to stop importing soxr at the top, this fails and it can go
    r = _run("from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService\n")
    assert r.returncode != 0 and "soxr" in r.stderr


def test_resampling_on_the_stand_in_fails_and_says_why():
    r = _run("""
        import realtime_harness, soxr
        for call in (lambda: soxr.resample(b"", 16000, 24000), lambda: soxr.ResampleStream(16000, 24000, 1)):
            try:
                call()
            except RuntimeError as e:
                assert "no resampler" in str(e) and "24 kHz" in str(e), e
            else:
                raise SystemExit("resampled on a stand-in")
        print("refused")
    """)
    assert r.returncode == 0 and r.stdout.strip() == "refused", r.stderr[-2000:]


def test_24_khz_audio_passes_pipecats_resampler_untouched_on_the_stand_in():
    r = _run("""
        import asyncio, realtime_harness
        from pipecat.audio.utils import create_stream_resampler
        audio = bytes(960)
        assert asyncio.run(create_stream_resampler().resample(audio, 24000, 24000)) is audio
        print("untouched")
    """)
    assert r.returncode == 0 and r.stdout.strip().endswith("untouched"), r.stderr[-2000:]


def test_an_installed_soxr_is_left_alone(monkeypatch):
    import types

    from realtime_harness import standins
    real = types.ModuleType("soxr")
    monkeypatch.setitem(sys.modules, "soxr", real)
    assert standins.install() is False and sys.modules["soxr"] is real
