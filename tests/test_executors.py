"""Executors must fail loudly when their tool is missing.

The compiler is the one that matters: a compile step that quietly does nothing
while reporting success ships exactly the readable source the binary-only
requirement exists to prevent.
"""
import os
import tarfile

import pytest

from studio import executors as ex

NUITKA_PRESENT = ex.tool_available("nuitka") or ex._module_present("nuitka")


# -- a missing compiler FAILS, never skips silently --------------------------

@pytest.mark.skipif(NUITKA_PRESENT,
                    reason="nuitka is installed here, so the missing-compiler path "
                           "cannot be exercised on this machine")
def test_a_missing_compiler_fails_and_explains_the_consequence(tmp_path):
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(tmp_path / "out"))()
    assert ok is False, detail
    assert "readable .py" in detail, detail


# -- a missing scanner is an error, not an empty result ----------------------

@pytest.mark.skipif(ex.tool_available("grype"), reason="grype is installed here")
def test_a_missing_scanner_raises_and_offers_the_honest_alternative():
    with pytest.raises(ex.ExecutorError) as e:
        ex.grype_scanner("some:image")()
    assert "run_sca: false" in str(e.value), str(e.value)


# -- bundler refuses when the script is not in the repo ----------------------

def test_the_bundler_refuses_without_the_product_script(tmp_path):
    ok, path, detail = ex.bundle_builder(str(tmp_path), str(tmp_path), "3.24")("full")
    assert ok is False, detail
    assert "product repository" in detail, detail


# -- tar verification --------------------------------------------------------

@pytest.fixture
def good_tar(tmp_path):
    inner = tmp_path / "app.txt"
    inner.write_text("x")
    path = tmp_path / "good.tar"
    with tarfile.open(path, "w") as tf:
        tf.add(str(inner), arcname="images/app.tar")
    return str(path)


def test_a_valid_tar_passes(good_tar):
    ok, detail = ex.tar_verifier()(good_tar)
    assert ok, detail


def test_an_expected_entry_is_found(good_tar):
    ok, detail = ex.tar_verifier(["images/app.tar"])(good_tar)
    assert ok, detail


def test_a_missing_expected_entry_fails(good_tar):
    """Catching this here saves another multi-gigabyte transfer to the site."""
    ok, detail = ex.tar_verifier(["images/llm.tar"])(good_tar)
    assert not ok and "missing" in detail, detail


def test_a_corrupt_tar_fails(tmp_path):
    bad = tmp_path / "bad.tar"
    bad.write_bytes(b"not a tar at all")
    ok, detail = ex.tar_verifier()(str(bad))
    assert not ok, detail


def test_an_absent_file_fails(tmp_path):
    ok, detail = ex.tar_verifier()(str(tmp_path / "nope.tar"))
    assert not ok, detail


# -- publisher refuses without credentials, never pretends -------------------

def test_the_publisher_refuses_without_credentials(monkeypatch, good_tar):
    monkeypatch.delenv("ARTIFACTORY_USER", raising=False)
    monkeypatch.delenv("ARTIFACTORY_TOKEN", raising=False)
    ok, detail = ex.artifactory_publisher(
        "https://art.example.com", "auditbox/releases")(good_tar, "3.24")
    assert ok is False, detail
    assert "environment" in detail, detail


# -- a missing binary raises a clear error, not a traceback ------------------

def test_an_unknown_binary_raises_a_clear_error():
    with pytest.raises(ex.ExecutorError) as e:
        ex._run(["definitely-not-a-real-binary-xyz"])
    assert "not installed" in str(e.value), str(e.value)


def test_tool_detection():
    assert not ex.tool_available("definitely-not-real-xyz")


# -- compilation happens in the image the product ships on -------------------
# It used to run nuitka on the build machine. On a Windows host that produced
# a .pyd for cp314-win_amd64 -- the wrong platform and the wrong interpreter for
# a python:3.11-slim Linux image, an artifact that could never be shipped. Built
# there anyway it compiled 54 modules successfully and then segfaulted on
# import. Built inside the target image the same source imports cleanly, so the
# host was the problem, not the compiler.

def test_compilation_requires_docker(monkeypatch, tmp_path):
    """Without docker it fails, and says what turning it off would cost."""
    monkeypatch.setattr(ex, "tool_available", lambda name: False)
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(tmp_path / "out"))()
    assert ok is False
    assert "inside the image" in detail, detail
    assert "compile_source: false" in detail, detail


def test_a_build_that_does_not_import_is_refused(monkeypatch, tmp_path):
    """"It compiled" is not evidence it works.

    The Windows attempt compiled cleanly and died in exec_module. A module that
    fails at load is worse than shipping source: the customer's app will not
    start at all.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "src.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    monkeypatch.setattr(ex, "tool_available", lambda name: name == "docker")
    # docker exits 0 and produced a .so, but the import probe never printed its marker
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "compiled fine, no marker"))
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(out))()
    assert ok is False
    assert "does not import" in detail, detail


def test_a_successful_build_reports_the_image_and_its_data(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "src.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    # Nuitka compiles code, not data. A repo with assets is the real case.
    (tmp_path / "src" / "api" / "static").mkdir(parents=True)
    (tmp_path / "src" / "api" / "static" / "index.html").write_text("<html>")
    monkeypatch.setattr(ex, "tool_available", lambda name: name == "docker")
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "NUITKA_IMPORT_OK"))
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(out))()
    assert ok, detail
    assert "python:3.11-slim" in detail and "imported" in detail, detail
    assert "1 data file" in detail, detail
    assert (out / "src" / "api" / "static" / "index.html").exists(), "asset not staged"


def test_a_compile_with_no_data_files_is_refused(monkeypatch, tmp_path):
    """It imported, then died on RuntimeError: Directory 'src/api/static'
    does not exist. Compiling is not the same as being shippable."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "src.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "only.py").write_text("x = 1")
    monkeypatch.setattr(ex, "tool_available", lambda name: name == "docker")
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "NUITKA_IMPORT_OK"))
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(out))()
    assert ok is False
    assert "relative path" in detail, detail


def test_producing_nothing_is_a_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "tool_available", lambda name: name == "docker")
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "NUITKA_IMPORT_OK"))
    ok, detail = ex.nuitka_compiler(str(tmp_path), str(tmp_path / "empty"))()
    assert ok is False
    assert "no native modules" in detail, detail


# -- reading pytest's summary ------------------------------------------------
# Found by running a build through the UI: the tests gate reported "0 passed"
# for a suite of 258. The old parser split the line on whitespace and compared
# tokens to "passed", but pytest writes "passed," with a comma whenever it also
# reports skips, so the comparison never matched. Failures were still caught,
# via the non-zero exit code -- but "0 passed" is exactly how a run that
# collected nothing looks, which is the one thing this step exists to notice.

@pytest.mark.parametrize("line,passed,failed", [
    ("258 passed, 6 skipped in 14.81s", 258, 0),
    ("123 passed in 4.9s", 123, 0),
    ("3 failed, 255 passed in 15s", 255, 3),
    ("1 failed, 2 errors, 5 passed in 2s", 5, 3),
    ("6 skipped in 0.4s", 0, 0),
])
def test_the_summary_line_is_read_correctly(line, passed, failed):
    counts = ex._PYTEST_COUNT_RE.findall(line)
    got_p = sum(int(n) for n, w in counts if w == "passed")
    got_f = sum(int(n) for n, w in counts if w != "passed")
    assert (got_p, got_f) == (passed, failed), counts


def test_a_green_run_reports_its_real_count(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "...\n258 passed, 6 skipped in 14.8s"))
    passed, failed, _ = ex.pytest_runner(str(tmp_path))()
    assert (passed, failed) == (258, 0)


def test_failures_are_counted_not_just_inferred(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (1, "...\n3 failed, 255 passed in 15s"))
    passed, failed, _ = ex.pytest_runner(str(tmp_path))()
    assert failed == 3 and passed == 255


def test_a_green_exit_with_no_readable_count_is_not_a_pass(monkeypatch, tmp_path):
    """Exiting 0 and counting nothing means the summary was not understood.

    Treating that as success is how "0 passed" slipped through as a green gate.
    """
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "something unexpected"))
    passed, failed, detail = ex.pytest_runner(str(tmp_path))()
    assert failed == 1
    assert "no test count" in detail, detail


def test_collecting_nothing_still_fails_distinguishably(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (5, "no tests ran in 0.10s"))
    passed, failed, detail = ex.pytest_runner(str(tmp_path))()
    assert failed == 1
    assert "collected no tests" in detail, detail
