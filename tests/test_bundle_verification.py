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


# -- the verifying key must be in the repo before anything is built ----------
# Found on a real inspection: config/licence_public.pem did not exist, was not
# tracked, and was absent from the built app image, because .gitignore carried a
# blanket *.pem. An installation with AUDITBOX_ENFORCE_ENTITLEMENTS=1 then finds
# no key, fails closed, and refuses every framework -- the customer gets a
# product that starts and audits nothing. Failing closed is right at runtime;
# catching it before the bundle ships is this gate's job.

def _repo_with_key(tmp_path, body):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "licence_public.pem").write_text(body)
    return str(tmp_path)


def test_a_repo_without_the_verifying_key_is_refused(tmp_path):
    ok, detail = ex.licence_key_present(str(tmp_path))()
    assert not ok
    assert "studio keygen" in detail, detail


def test_a_real_public_key_passes(tmp_path):
    repo = _repo_with_key(tmp_path,
        "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA\n-----END PUBLIC KEY-----\n")
    ok, detail = ex.licence_key_present(repo)()
    assert ok, detail
    assert "public half only" in detail


def test_a_private_key_in_that_slot_is_refused(tmp_path):
    """Shipping the signing key would let any customer mint their own licence."""
    repo = _repo_with_key(tmp_path,
        "-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIA\n-----END PRIVATE KEY-----\n")
    ok, detail = ex.licence_key_present(repo)()
    assert not ok
    assert "not a public key" in detail or "PRIVATE" in detail, detail


def test_junk_in_that_slot_is_refused(tmp_path):
    ok, _ = ex.licence_key_present(_repo_with_key(tmp_path, "not a key at all"))()
    assert not ok


def test_the_step_stops_the_build(tmp_path):
    r = pl.step_licence_key(ex.licence_key_present(str(tmp_path)))
    assert not r.ok and r.name == "licence key"


# -- a git tag is not a filename ---------------------------------------------
# Spotted the moment the page showed what a bundle would contain: passing the
# tag v3.25 straight through produced "INSTALL_vv3.25.md" and a directory named
# AICyberAuditBox-v3.25, while every release this product has ever shipped is
# named AICyberAuditBox-3.23. The tag keeps its prefix for git; anything that
# becomes a name loses it.

@pytest.mark.parametrize("given,expected", [
    ("v3.25", "3.25"),
    ("3.25", "3.25"),
    ("v3.24.1", "3.24.1"),
    ("vnext", "vnext"),        # not a version; left alone rather than mangled
    ("", ""),
])
def test_the_filename_form_of_a_version(given, expected):
    assert ex.artifact_version(given) == expected


def test_expectations_use_the_filename_form():
    names = ex.bundle_expectations("full", "v3.25")
    assert "aicyberauditbox-images-3.25.tar" in names
    assert "INSTALL_v3.25.md" in names
    assert not any("vv3.25" in n for n in names), names
    assert not any("-v3.25" in n for n in names), names


def test_a_tagged_and_untagged_version_expect_the_same_files():
    assert ex.bundle_expectations("full", "v3.25") == ex.bundle_expectations("full", "3.25")
    assert ex.bundle_expectations("patch", "v3.25") == ex.bundle_expectations("patch", "3.25")


# -- what is inside the one big line -----------------------------------------
# The contents list showed aicyberauditbox-images-<v>.tar as a single entry.
# True, and it hides everything an operator wants to confirm is going: asked
# where the LLM, the database and the Python libraries were, the honest answer
# was "inside that one file". docker save writes five images into it.

def test_the_images_tar_is_broken_out():
    imgs = ex.bundle_images("v3.25", "3.10")
    tags = [i["tag"] for i in imgs]
    assert "aicyberauditbox-app:3.25" in tags
    assert "aicyberauditbox-llm:3.25" in tags
    assert "aicyberauditbox-llm-embed:3.25" in tags
    assert "aicyberauditbox-shakthidb:3.10" in tags
    assert "redis:7-alpine" in tags
    assert len(imgs) == 5


def test_each_image_says_what_it_holds():
    """The point is answering "where are the models", not listing tags."""
    for i in ex.bundle_images("3.25"):
        assert i["holds"], i
    joined = " ".join(i["holds"] for i in ex.bundle_images("3.25"))
    for thing in ("Python librar", "model weights", "PostgreSQL", "embedding"):
        assert thing in joined, thing


def test_the_images_use_the_filename_form_of_the_version():
    assert ex.bundle_images("v3.25")[0]["tag"] == "aicyberauditbox-app:3.25"


def test_the_database_version_is_read_from_the_product_not_guessed(tmp_path):
    """The db image is the one tag that does not follow the product version.

    Guessing produced shakthidb:1.0 for an image that is actually 3.10 -- a name
    the verification step would then look for and never find.
    """
    (tmp_path / "build_customer_bundle.py").write_text(
        'DB_VER = "9.9"\nDB_TAG = "aicyberauditbox-shakthidb:" + DB_VER\n')
    assert ex.product_db_version(str(tmp_path)) == "9.9"


def test_a_missing_bundler_falls_back_rather_than_crashing(tmp_path):
    assert ex.product_db_version(str(tmp_path)) == "3.10"


# -- the bundle is looked for where it was told to be written ----------------
# The studio never passed --out, so the bundler used its own default
# (<parent of repo>/customer_deployment_package/v<version>) while this searched
# the studio's out/ and then fell back to the newest .tar in the repo root. That
# fallback would have picked up a stale tar from an earlier release, verified it,
# signed a licence for it and published it as the new build.

def test_the_bundler_is_told_where_to_write(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "build_customer_bundle.py").write_text("# stub\n")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return 0, "built"

    monkeypatch.setattr(ex, "_run", fake_run)
    ex.bundle_builder(str(repo), str(tmp_path / "out"), "v3.25")("full")
    assert "--out" in seen["cmd"], seen["cmd"]
    assert "--version" in seen["cmd"]
    # The filename form, so the bundler names things as every release has been.
    assert "3.25" in seen["cmd"] and "v3.25" not in seen["cmd"], seen["cmd"]


def test_a_stale_tar_elsewhere_is_never_picked_up(monkeypatch, tmp_path):
    """The bug this guards: an old tar in the repo becoming "the new bundle"."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "build_customer_bundle.py").write_text("# stub\n")
    (repo / "AICyberAuditBox-3.20-complete.tar").write_bytes(b"an old release")
    out = tmp_path / "out"
    out.mkdir()
    (out / "something-old.tar").write_bytes(b"also not this build")
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "built"))

    ok, path, detail = ex.bundle_builder(str(repo), str(out), "v3.25")("full")
    assert ok is False, "a tar from somewhere else was accepted as the build"
    assert path is None
    assert "no tar was found" in detail, detail


def test_the_bundle_found_in_the_right_place_is_accepted(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "build_customer_bundle.py").write_text("# stub\n")
    out = tmp_path / "out"
    (out / "v3.25").mkdir(parents=True)
    real = out / "v3.25" / "AICyberAuditBox-3.25-complete.tar"
    real.write_bytes(b"this build")
    monkeypatch.setattr(ex, "_run", lambda *a, **k: (0, "built"))

    ok, path, detail = ex.bundle_builder(str(repo), str(out), "v3.25")("full")
    assert ok, detail
    assert os.path.basename(path) == "AICyberAuditBox-3.25-complete.tar"


def test_a_patch_base_also_uses_the_filename_form(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "build_customer_bundle.py").write_text("# stub\n")
    seen = {}
    monkeypatch.setattr(ex, "_run", lambda cmd, **k: (seen.update(cmd=cmd), (0, "x"))[1])
    ex.bundle_builder(str(repo), str(tmp_path / "o"), "v3.25", patch_from="v3.24")("patch")
    assert "3.24" in seen["cmd"] and "v3.24" not in seen["cmd"], seen["cmd"]
