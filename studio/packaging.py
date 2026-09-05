"""Encrypting a bundle, and deciding whether a patch is legal.

Two jobs that both protect a delivery:

  encrypt_bundle / decrypt_bundle
      The tar is encrypted with a key derived from the licence key, so a bundle
      only opens on the install it was issued for. Copying a customer's tar to
      another site gives an unopenable file.

      Be clear about what this does and does not do. The installer must decrypt
      to run, so the material to do so is present at the customer site -- this
      stops casual copying between sites, not a determined reverse engineer.
      Source protection comes from compiling the code, not from this.

  patch_is_legal
      A patch ships src/ and config/ only, rebuilt on the image the customer
      already runs. That is invalid the moment a lower layer moved: new Python
      dependencies, a different model, a changed Dockerfile. The bundler's own
      docstring says so, and a human forgets. Checked here instead.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MAGIC = b"ABXBUNDLE1"          # so a wrong file fails clearly, not confusingly
_SALT_LEN = 16
_NONCE_LEN = 12
_KDF_ITERATIONS = 200_000
_CHUNK = 4 * 1024 * 1024


class PackagingError(Exception):
    pass


# ── encryption ───────────────────────────────────────────────────────────────

def derive_key(licence_key: str, salt: bytes) -> bytes:
    """32-byte AES key from the licence key.

    PBKDF2-SHA256. The licence key is long and random, so iteration count is not
    load-bearing the way it is for a human password, but it costs nothing and
    protects the case where someone issues a short one.
    """
    if not licence_key:
        raise PackagingError("a licence key is required to derive the bundle key")
    return hashlib.pbkdf2_hmac("sha256", licence_key.encode(), salt, _KDF_ITERATIONS, dklen=32)


def encrypt_bundle(src_path: str, dst_path: str, licence_key: str) -> dict:
    """Encrypt a bundle tar. Returns a manifest of what was written.

    AES-256-GCM, which authenticates as well as encrypts: a bundle altered in
    transit fails to decrypt rather than silently installing modified software.
    """
    if not os.path.isfile(src_path):
        raise PackagingError(f"no bundle to encrypt at {src_path}")
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = derive_key(licence_key, salt)

    with open(src_path, "rb") as fh:
        plaintext = fh.read()
    sha_plain = hashlib.sha256(plaintext).hexdigest()
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _MAGIC)

    with open(dst_path, "wb") as out:
        out.write(_MAGIC)
        out.write(salt)
        out.write(nonce)
        out.write(ciphertext)

    return {
        "source": os.path.basename(src_path),
        "output": os.path.basename(dst_path),
        "plaintext_bytes": len(plaintext),
        "encrypted_bytes": os.path.getsize(dst_path),
        "sha256_plaintext": sha_plain,
        "sha256_encrypted": _sha256_file(dst_path),
    }


def decrypt_bundle(src_path: str, dst_path: str, licence_key: str) -> dict:
    """Decrypt a bundle. Raises if the licence is wrong or the file was altered."""
    with open(src_path, "rb") as fh:
        blob = fh.read()
    head = len(_MAGIC) + _SALT_LEN + _NONCE_LEN
    if len(blob) < head or not blob.startswith(_MAGIC):
        raise PackagingError("not an AuditBox bundle, or the file is truncated")

    salt = blob[len(_MAGIC):len(_MAGIC) + _SALT_LEN]
    nonce = blob[len(_MAGIC) + _SALT_LEN:head]
    key = derive_key(licence_key, salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, blob[head:], _MAGIC)
    except Exception as exc:
        raise PackagingError(
            "bundle could not be decrypted: the licence key does not match the "
            "one it was built for, or the file was modified"
        ) from exc

    with open(dst_path, "wb") as out:
        out.write(plaintext)
    return {"output": os.path.basename(dst_path), "bytes": len(plaintext),
            "sha256": hashlib.sha256(plaintext).hexdigest()}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# ── patch legality ───────────────────────────────────────────────────────────

# Touch any of these between two versions and a patch cannot apply: they change
# layers beneath src/, which is all a patch replaces.
LOWER_LAYER_PATHS = (
    "requirements.txt",
    "requirements.lock.txt",
    "Dockerfile",
    "Dockerfile.app",
    "Dockerfile.llm",
    "docker-compose.yml",
    "docker-compose.customer.yml",
)


@dataclass(frozen=True)
class PatchDecision:
    legal: bool
    shape: str                  # "patch" | "full"
    blocking_changes: List[str]
    reason: str


def patch_is_legal(repo_path: str, from_ref: str, to_ref: str,
                   model_changed: bool = False) -> PatchDecision:
    """Decide whether a patch from from_ref to to_ref can apply.

    Asks git what changed. A patch replaces src/ and config/ on top of the image
    the customer already runs, so if the Python dependencies or any Dockerfile
    moved, the layers underneath differ and the patch would either fail to build
    or produce an image that is not what was tested.
    """
    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"{from_ref}..{to_ref}"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackagingError(f"could not diff {from_ref}..{to_ref}: {exc}") from exc
    if out.returncode != 0:
        raise PackagingError(
            f"git could not compare {from_ref}..{to_ref}: {out.stderr.strip()[:200]}"
        )

    changed = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    blocking = sorted({p for p in changed if p in LOWER_LAYER_PATHS})
    if model_changed:
        blocking.append("<model file>")

    if blocking:
        return PatchDecision(
            legal=False, shape="full", blocking_changes=blocking,
            reason=("a patch rebuilds only src/ and config/ on the customer's existing "
                    "image; these changed beneath that layer: " + ", ".join(blocking)),
        )
    if not changed:
        return PatchDecision(True, "patch", [],
                             f"nothing changed between {from_ref} and {to_ref}")
    return PatchDecision(
        legal=True, shape="patch", blocking_changes=[],
        reason=f"{len(changed)} file(s) changed, none beneath the app layer",
    )
