"""Encryption must fail closed, and an illegal patch must be refused."""
import os
import subprocess

import pytest

from studio.packaging import (
    encrypt_bundle, decrypt_bundle, derive_key, patch_is_legal,
    PackagingError, LOWER_LAYER_PATHS,
)

LIC = "AUDITBOX-LIC-1.abc.def"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A plaintext bundle and its encrypted form, built once for the module."""
    tmp = tmp_path_factory.mktemp("pack")
    plain = tmp / "bundle.tar"
    body = os.urandom(300_000)
    plain.write_bytes(body)
    enc = tmp / "bundle.enc"
    manifest = encrypt_bundle(str(plain), str(enc), LIC)
    return dict(tmp=tmp, plain=str(plain), body=body, enc=str(enc), manifest=manifest)


# -- round trip --------------------------------------------------------------

def test_round_trip(bundle):
    assert os.path.getsize(bundle["enc"]) > 0
    with open(bundle["enc"], "rb") as fh:
        assert fh.read()[:64] != bundle["body"][:64], "ciphertext matches plaintext"
    back = str(bundle["tmp"] / "back.tar")
    res = decrypt_bundle(bundle["enc"], back, LIC)
    with open(back, "rb") as fh:
        assert fh.read() == bundle["body"]
    assert res["sha256"] == bundle["manifest"]["sha256_plaintext"]


# -- the wrong licence cannot open it ----------------------------------------

def test_the_wrong_licence_cannot_open_it(bundle):
    with pytest.raises(PackagingError) as e:
        decrypt_bundle(bundle["enc"], str(bundle["tmp"] / "x.tar"),
                       "AUDITBOX-LIC-1.other.key")
    assert "licence" in str(e.value).lower()


# -- tampering is detected, not silently installed ---------------------------

def test_a_modified_bundle_is_rejected(bundle):
    with open(bundle["enc"], "rb") as fh:
        blob = bytearray(fh.read())
    blob[-1] ^= 0xFF
    bad = bundle["tmp"] / "tampered.enc"
    bad.write_bytes(bytes(blob))
    with pytest.raises(PackagingError):
        decrypt_bundle(str(bad), str(bundle["tmp"] / "y.tar"), LIC)


def test_a_foreign_file_fails_clearly(bundle):
    junk = bundle["tmp"] / "junk.bin"
    junk.write_bytes(b"not a bundle at all")
    with pytest.raises(PackagingError) as e:
        decrypt_bundle(str(junk), str(bundle["tmp"] / "z.tar"), LIC)
    assert "bundle" in str(e.value).lower()


# -- key derivation ----------------------------------------------------------

def test_key_derivation_is_deterministic_and_salted():
    salt = os.urandom(16)
    assert derive_key(LIC, salt) == derive_key(LIC, salt)
    assert derive_key(LIC, salt) != derive_key(LIC, os.urandom(16))
    assert derive_key(LIC, salt) != derive_key("other", salt)
    assert len(derive_key(LIC, salt)) == 32


def test_an_empty_licence_is_refused():
    with pytest.raises(PackagingError):
        derive_key("", os.urandom(16))


def test_the_salt_is_random_per_encryption(bundle):
    """Encrypting the same bundle twice must not produce the same header.

    The original of this test compared a file's bytes against the dict that
    encrypt_bundle returns, which is never equal and so always passed. It now
    compares the two salts.
    """
    second = str(bundle["tmp"] / "e2.enc")
    encrypt_bundle(bundle["plain"], second, LIC)
    with open(bundle["enc"], "rb") as a, open(second, "rb") as b:
        assert a.read()[10:26] != b.read()[10:26]


# -- patch legality, against a real git repo ---------------------------------

@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """Three tagged versions: v1, v2 (code only), v3 (dependency bump)."""
    path = tmp_path_factory.mktemp("repo")

    def git(*a):
        return subprocess.run(["git", "-C", str(path), *a],
                              capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (path / "src").mkdir()
    (path / "src" / "app.py").write_text("v1")
    (path / "requirements.txt").write_text("fastapi==1\n")
    git("add", "-A"); git("commit", "-qm", "v1"); git("tag", "v1")

    (path / "src" / "app.py").write_text("v2 code only")
    git("add", "-A"); git("commit", "-qm", "code change"); git("tag", "v2")

    (path / "requirements.txt").write_text("fastapi==2\n")
    git("add", "-A"); git("commit", "-qm", "dep bump"); git("tag", "v3")
    return str(path)


def test_a_code_only_change_allows_a_patch(repo):
    d = patch_is_legal(repo, "v1", "v2")
    assert d.legal and d.shape == "patch", d.reason


def test_a_dependency_change_refuses_a_patch(repo):
    """A patch cannot rebuild the layer the dependency lives in."""
    d = patch_is_legal(repo, "v2", "v3")
    assert not d.legal and d.shape == "full", d.reason
    assert "requirements.txt" in d.blocking_changes


def test_spanning_versions_still_catches_it(repo):
    assert not patch_is_legal(repo, "v1", "v3").legal


def test_no_change_is_legal_and_says_so(repo):
    d = patch_is_legal(repo, "v2", "v2")
    assert d.legal and "nothing changed" in d.reason


def test_a_changed_model_blocks_a_patch(repo):
    assert not patch_is_legal(repo, "v1", "v2", model_changed=True).legal


def test_unknown_refs_fail_loudly(repo):
    """Never default to 'legal' when the comparison could not be made."""
    with pytest.raises(PackagingError):
        patch_is_legal(repo, "v1", "does-not-exist")


def test_every_lower_layer_path_is_a_real_concern():
    assert all(p in ("requirements.txt", "requirements.lock.txt", "Dockerfile",
                     "Dockerfile.app", "Dockerfile.llm", "docker-compose.yml",
                     "docker-compose.customer.yml")
               for p in LOWER_LAYER_PATHS)
