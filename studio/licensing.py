"""Licence keys that carry entitlements, signed so they cannot be edited.

The product's existing licence is a token wallet with an expiry: it counts
tokens and dates, and knows nothing about which frameworks were bought. That is
why a PQC-only customer cannot be served today -- there is no field to put the
answer in, and the audit path never asks.

A licence here is a signed payload:

    {customer, expires, frameworks: ["PQC"], seats, issued, licence_id}

Ed25519 rather than a shared secret, deliberately. A symmetric scheme would
require the verifying key to ship inside the product, where any customer can
read it and mint their own licence granting every framework. With Ed25519 the
product carries only the PUBLIC key, which verifies signatures and cannot
create them. The private key never leaves the build machine.

That asymmetry is the whole point: the thing shipped to customers is incapable
of forging what it verifies.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

_SCHEME = "AUDITBOX-LIC-1"


class LicenceError(Exception):
    """Raised for a licence that is malformed, unsigned, forged or expired."""


@dataclass(frozen=True)
class VerifiedLicence:
    customer: str
    licence_id: str
    frameworks: List[str]
    expires: date
    seats: int
    issued: datetime
    tokens: Optional[int]

    def permits(self, framework: str) -> bool:
        """Whether this licence covers the named framework.

        Compared case-insensitively because the caller may pass a UI label or a
        stored value; an entitlement check that fails on capitalisation would
        deny a paying customer.
        """
        want = str(framework or "").strip().upper()
        return any(want == f.strip().upper() for f in self.frameworks)

    @property
    def days_remaining(self) -> int:
        return (self.expires - date.today()).days

    @property
    def expired(self) -> bool:
        return self.days_remaining < 0


# ── key management ───────────────────────────────────────────────────────────

def generate_keypair() -> tuple:
    """A fresh signing keypair. Returns (private_pem, public_pem) as bytes."""
    private = Ed25519PrivateKey.generate()
    priv_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def load_private_key(pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise LicenceError("signing key is not Ed25519")
    return key


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise LicenceError("verifying key is not Ed25519")
    return key


def private_key_from_env(var: str = "AUDITBOX_LICENCE_KEY") -> Ed25519PrivateKey:
    """Read the signing key from the environment, never from the repository.

    A signing key committed once stays in git history forever, and anyone
    holding it can mint a licence granting every framework -- including the
    PQC-only customer granting themselves the rest of the product.
    """
    pem = os.environ.get(var, "").strip()
    if not pem:
        raise LicenceError(
            f"{var} is not set. The licence signing key is read from the "
            f"environment or a secret store, never from a file in the repo."
        )
    return load_private_key(pem.encode() if isinstance(pem, str) else pem)


# ── issue / verify ───────────────────────────────────────────────────────────

def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64u(txt: str) -> bytes:
    pad = "=" * (-len(txt) % 4)
    return base64.urlsafe_b64decode(txt + pad)


def issue(
    private_key: Ed25519PrivateKey,
    *,
    customer: str,
    expires: date,
    frameworks: List[str],
    seats: int = 5,
    tokens: Optional[int] = None,
    licence_id: Optional[str] = None,
    issued: Optional[datetime] = None,
) -> str:
    """Produce a signed licence key.

    Format:  AUDITBOX-LIC-1.<payload>.<signature>, both base64url.
    The payload is readable by design -- a customer may confirm what they bought
    and when it lapses. It is signed, not secret; the protection is that it
    cannot be altered, not that it cannot be read.
    """
    if not customer or not str(customer).strip():
        raise LicenceError("customer is required")
    if not frameworks:
        raise LicenceError("a licence with no frameworks grants nothing")
    payload = {
        "v": 1,
        "customer": str(customer).strip(),
        "licence_id": licence_id or _b64u(os.urandom(9)),
        "frameworks": [str(f).strip().upper() for f in frameworks],
        "expires": expires.isoformat(),
        "seats": int(seats),
        "issued": (issued or datetime.now(timezone.utc)).isoformat(),
    }
    if tokens is not None:
        payload["tokens"] = int(tokens)

    # sort_keys so the same inputs always produce the same bytes to sign;
    # separators so no incidental whitespace changes the signature.
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = private_key.sign(body)
    return f"{_SCHEME}.{_b64u(body)}.{_b64u(sig)}"


def verify(public_key: Ed25519PublicKey, licence_key: str,
           *, check_expiry: bool = True) -> VerifiedLicence:
    """Verify and decode a licence key. Raises LicenceError if it is not good.

    Signature first, then expiry: an expired licence and a forged one are
    different problems and the caller should be able to tell them apart, but
    neither is ever trusted.
    """
    if not licence_key or not isinstance(licence_key, str):
        raise LicenceError("no licence key supplied")
    parts = licence_key.strip().split(".")
    if len(parts) != 3 or parts[0] != _SCHEME:
        raise LicenceError("licence key is malformed or of an unknown scheme")

    try:
        body = _unb64u(parts[1])
        sig = _unb64u(parts[2])
    except Exception as exc:
        raise LicenceError(f"licence key is not decodable: {exc}") from exc

    try:
        public_key.verify(sig, body)
    except InvalidSignature as exc:
        raise LicenceError(
            "licence signature is invalid -- the key was altered or was not "
            "issued by this vendor"
        ) from exc

    try:
        payload = json.loads(body)
        lic = VerifiedLicence(
            customer=payload["customer"],
            licence_id=payload["licence_id"],
            frameworks=list(payload["frameworks"]),
            expires=date.fromisoformat(payload["expires"]),
            seats=int(payload["seats"]),
            issued=datetime.fromisoformat(payload["issued"]),
            tokens=payload.get("tokens"),
        )
    except (KeyError, ValueError) as exc:
        raise LicenceError(f"licence payload is incomplete: {exc}") from exc

    if check_expiry and lic.expired:
        raise LicenceError(
            f"licence expired on {lic.expires.isoformat()} "
            f"({abs(lic.days_remaining)} days ago)"
        )
    return lic
