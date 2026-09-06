"""The CLI's output has to survive a Windows console.

The operator running a release is on cp1252 more often than not. Typographic
characters in the printed strings -- an arrow in the invalid-profile report, a
middot in the plan header -- either raised UnicodeEncodeError or degraded to a
literal "\u2192" in the middle of the message somebody non-technical has to
read to fix their profile.
"""
import ast
import os
import subprocess
import sys

import pytest

# A literal backslash-u, i.e. what an unprintable character degrades into.
ESCAPE_PREFIX = chr(92) + "u"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [os.path.join(ROOT, "studio", f)
           for f in sorted(os.listdir(os.path.join(ROOT, "studio")))
           if f.endswith(".py")]


def _string_literals(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield node.lineno, part.value


@pytest.mark.parametrize("path", SOURCES, ids=[os.path.basename(p) for p in SOURCES])
def test_no_string_literal_carries_a_character_a_cp1252_console_cannot_print(path):
    """Comments may use box drawing; anything that can reach stdout may not.

    Docstrings are exempt -- they are documentation, not output.
    """
    with open(path, encoding="utf-8") as fh:
        docstrings = {d for d in _all_docstrings(fh.read())}
    offenders = []
    for lineno, text in _string_literals(path):
        if text in docstrings:
            continue
        bad = sorted({c for c in text if ord(c) > 127})
        if bad:
            offenders.append("%s:%d %s" % (os.path.basename(path), lineno,
                                           " ".join("U+%04X" % ord(c) for c in bad)))
    assert not offenders, "non-ASCII in printable strings: " + "; ".join(offenders)


def _all_docstrings(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                yield doc


def _run_cli(args, tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    return subprocess.run([sys.executable, "-m", "studio.cli", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_an_invalid_profile_reports_its_fields_readably(tmp_path):
    """The worst case: the person who most needs this message reads it least well."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'licence:\n  customer: ""\n  expires: 2027-01-01\n  frameworks: []\n'
        'hardware:\n  physical_cores: 0\n  ram_gb: 0\n'
        'model: "google_gemma-4-E4B-it-Q4_K_M.gguf"\n')
    r = _run_cli(["validate", str(bad)], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "licence -> customer" in out, out
    assert ESCAPE_PREFIX not in out, "escape sequences leaked into the message: " + out
    assert "Traceback" not in out, out


def test_validate_prints_cleanly_on_a_cp1252_console(tmp_path):
    r = _run_cli(["validate", "profiles/stpi.yaml"], tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert ESCAPE_PREFIX not in out and "Traceback" not in out, out
