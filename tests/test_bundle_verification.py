"""Nothing may ship unverified: not the images, not the weights, not the bytes.

Three gaps, each of which would have reached a customer:

  the verify step was handed an empty expectation list, so it proved only that
  the tar opened -- a bundle with no images tar in it passed;

  no check ever looked inside the LLM image, so verify_images_tar confirming
  the tag was in the tar said nothing about whether /models was populated;

  nothing hashed the artifact, and a truncated multi-gigabyte transfer usually
  still opens as a tar.
"""
import hashlib
import os
import tarfile

import pytest

from studio import executors as ex
from studio import pipeline as pl
from tests.test_pipeline import make_profile


# -- what a bundle must contain ----------------------------------------------

def test_a_full_bundle_must_name_the_images_tar():
    names = ex.bundle_expectations("full", "3.24")
    assert "aicyberauditbox-images-3.24.tar" in names
    assert "docker-compose.yml" in names
    assert "install.sh" in names


def test_a_patch_bundle_expects_the_patch_pieces_instead():
    names = ex.bundle_expectations("patch", "3.24")
    assert "apply_patch.sh" in names
    assert "Dockerfile.app.rebase" in names
    assert not any("images" in n for n in names), names


def _bundle(tmp_path, entries):
    src = tmp_path / "content.txt"
    src.write_text("x")
    path = tmp_path / "bundle.tar"
    with tarfile.open(path, "w") as tf:
        for name in entries:
            tf.add(str(src), arcname=name)
    return str(path)


def test_a_bundle_without_the_images_tar_is_refused(tmp_path):
    """The gap exactly: this bundle used to pass and be published."""
    path = _bundle(tmp_path, ["AICyberAuditBox-3.24/docker-compose.yml",
                              "AICyberAuditBox-3.24/install.sh",
                              "AICyberAuditBox-3.24/install.bat",
                              "AICyberAuditBox-3.24/INSTALL_v3.24.md"])
    ok, detail = ex.tar_verifier(ex.bundle_expectations("full", "3.24"))(path)
    assert not ok, "a bundle with no images tar was accepted"
    assert "aicyberauditbox-images-3.24.tar" in detail, detail


def test_a_complete_bundle_passes(tmp_path):
    path = _bundle(tmp_path, ["AICyberAuditBox-3.24/aicyberauditbox-images-3.24.tar",
                              "AICyberAuditBox-3.24/docker-compose.yml",
                              "AICyberAuditBox-3.24/install.sh",
                              "AICyberAuditBox-3.24/install.bat",
                              "AICyberAuditBox-3.24/INSTALL_v3.24.md"])
    ok, detail = ex.tar_verifier(ex.bundle_expectations("full", "3.24"))(path)
    assert ok, detail


# -- are the weights actually in the image? ----------------------------------

GOOD_LISTING = (
    "total 17000000\n"
    "-rw-r--r-- 1 root root 12669647328 Sep  1 10:00 gemma-4-12B-it-Q8_0.gguf\n"
    "-rw-r--r-- 1 root root  5405168384 Sep  1 10:00 google_gemma-4-E4B-it-Q4_K_M.gguf\n"
    "-rw-r--r-- 1 root root   274290560 Sep  1 10:00 nomic-embed-text-v1.5.f16.gguf\n"
)


@pytest.fixture
def fake_docker(monkeypatch):
    """Stand in for the docker CLI so the check is testable without images."""
    def install(listing, code=0):
        monkeypatch.setattr(ex, "tool_available", lambda name: name == "docker")
        monkeypatch.setattr(ex, "_run", lambda *a, **k: (code, listing))
    return install


def test_a_populated_image_passes(fake_docker):
    fake_docker(GOOD_LISTING)
    ok, detail = ex.image_model_verifier("aicyberauditbox-llm:3.24")()
    assert ok, detail
    assert "3 model weight(s)" in detail


def test_an_image_with_no_weights_is_refused(fake_docker):
    """Starts fine, then fails at the customer's first inference."""
    fake_docker("total 0\n")
    ok, detail = ex.image_model_verifier("aicyberauditbox-llm:3.24")()
    assert not ok
    assert "gemma-4-12B-it-Q8_0.gguf" in detail
    assert "first inference" in detail, detail


def test_one_missing_weight_is_enough_to_refuse(fake_docker):
    fake_docker("\n".join(l for l in GOOD_LISTING.splitlines()
                          if "E4B" not in l) + "\n")
    ok, detail = ex.image_model_verifier("aicyberauditbox-llm:3.24")()
    assert not ok
    assert "google_gemma-4-E4B-it-Q4_K_M.gguf" in detail


def test_an_implausibly_small_weight_is_refused(fake_docker):
    """A 133-byte .gguf is a Git LFS pointer that got COPYed, not a model."""
    fake_docker(
        "-rw-r--r-- 1 root root 12669647328 Sep  1 10:00 gemma-4-12B-it-Q8_0.gguf\n"
        "-rw-r--r-- 1 root root         133 Sep  1 10:00 google_gemma-4-E4B-it-Q4_K_M.gguf\n"
        "-rw-r--r-- 1 root root   274290560 Sep  1 10:00 nomic-embed-text-v1.5.f16.gguf\n")
    ok, detail = ex.image_model_verifier("aicyberauditbox-llm:3.24")()
    assert not ok
    assert "implausibly small" in detail, detail


def test_a_missing_docker_raises_rather_than_passing(monkeypatch):
    """Never conclude the weights are fine because the check could not run."""
    monkeypatch.setattr(ex, "tool_available", lambda name: False)
    with pytest.raises(ex.ExecutorError) as e:
        ex.image_model_verifier("aicyberauditbox-llm:3.24")()
    assert "verify_models: false" in str(e.value)


def test_turning_the_model_check_off_is_flagged_loudly():
    r = pl.step_verify_models(make_profile(build=dict(verify_models=False)),
                              lambda: (True, ""))
    assert r.skipped and "WITHOUT WEIGHTS" in r.detail, r.detail


# -- integrity of the artifact -----------------------------------------------

def test_the_checksum_matches_the_file_and_is_written_beside_it(tmp_path):
    art = tmp_path / "bundle.tar"
    body = os.urandom(200_000)
    art.write_bytes(body)
    ok, digest, detail = ex.checksum_writer()(str(art))
    assert ok, detail
    assert digest == hashlib.sha256(body).hexdigest()
    side = tmp_path / "bundle.tar.sha256"
    assert side.exists()
    assert digest in side.read_text()
    assert "bundle.tar" in side.read_text()


def test_a_missing_artifact_does_not_produce_a_checksum(tmp_path):
    ok, digest, detail = ex.checksum_writer()(str(tmp_path / "nope.tar"))
    assert not ok and digest is None


def test_the_report_carries_the_digest(tmp_path):
    art = tmp_path / "artifact.tar"          # not __file__: that wrote a .sha256 into the repo
    art.write_bytes(b"payload")
    r = pl.step_checksum(make_profile(), str(art), ex.checksum_writer())
    assert r.ok and len(r.data["sha256"]) == 64
