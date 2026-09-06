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
