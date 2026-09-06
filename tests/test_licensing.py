"""Licence keys must resist a customer editing their own entitlements.

The product's existing licence is a token wallet with an expiry -- no field
says which frameworks were bought, so a PQC-only customer cannot be served.
These keys carry that, signed, so the answer cannot be changed by the party it
constrains.
"""
import base64
import datetime
import json
from datetime import date, timedelta

import pytest
from cryptography.hazmat.primitives import serialization

from studio.licensing import (
    generate_keypair, load_private_key, load_public_key, issue, verify,
    LicenceError, private_key_from_env,
)

FUTURE = date.today() + timedelta(days=365)


@pytest.fixture(scope="module")
def keypair():
    """One keypair for the module: generation is the slow part, not the tests."""
    priv_pem, pub_pem = generate_keypair()
    return dict(priv_pem=priv_pem, pub_pem=pub_pem,
                priv=load_private_key(priv_pem), pub=load_public_key(pub_pem))


@pytest.fixture(scope="module")
def priv(keypair):
    return keypair["priv"]


@pytest.fixture(scope="module")
def pub(keypair):
    return keypair["pub"]


@pytest.fixture(scope="module")
def pqc_key(priv):
    return issue(priv, customer="STPI", expires=FUTURE, frameworks=["PQC"])


def _reforge(key, **changes):
    """Edit the payload and keep the original signature -- what a customer would try."""
    head, body_b64, sig_b64 = key.split(".")
    pad = "=" * (-len(body_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(body_b64 + pad))
    payload.update(changes)
    body = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    return head + "." + body + "." + sig_b64


# -- a PQC-only licence grants PQC and nothing else --------------------------

def test_pqc_licence_grants_pqc_and_nothing_else(pub, pqc_key):
    lic = verify(pub, pqc_key)
    assert lic.customer == "STPI"
    assert lic.permits("PQC")
    assert not lic.permits("ISO27001")
    assert not lic.permits("VAPT")
    assert lic.permits("pqc") and lic.permits(" PQC ")
    assert lic.expires == FUTURE
    assert len(lic.licence_id) >= 8


# -- tampering is detected ---------------------------------------------------

def test_edited_entitlements_are_rejected(pub, pqc_key):
    """The customer grants themselves everything; the signature disagrees."""
    with pytest.raises(LicenceError) as e:
        verify(pub, _reforge(pqc_key, frameworks=["PQC", "ISO27001", "VAPT"]))
    assert "signature" in str(e.value).lower()


def test_an_extended_expiry_is_rejected(pub, pqc_key):
    forged = _reforge(pqc_key, expires=(date.today() + timedelta(days=99999)).isoformat())
    with pytest.raises(LicenceError):
        verify(pub, forged)


# -- a different vendor's key cannot sign for us -----------------------------

def test_a_foreign_signature_is_rejected(pub):
    other_priv, _ = generate_keypair()
    other = issue(load_private_key(other_priv), customer="STPI",
                  expires=FUTURE, frameworks=["PQC"])
    with pytest.raises(LicenceError):
        verify(pub, other)


# -- expiry is enforced, and distinguishable from forgery --------------------

@pytest.fixture(scope="module")
def expired_key(priv):
    return issue(priv, customer="X", expires=date.today() - timedelta(days=1),
                 frameworks=["PQC"])


def test_an_expired_licence_is_rejected(pub, expired_key):
    with pytest.raises(LicenceError) as e:
        verify(pub, expired_key)
    assert "expired" in str(e.value).lower()


def test_an_expired_licence_is_still_decodable_for_a_clear_message(pub, expired_key):
    """Expiry and forgery must not look the same to whoever reads the error."""
    lic = verify(pub, expired_key, check_expiry=False)
    assert lic.customer == "X"
    assert lic.expired and lic.days_remaining < 0


# -- malformed input never crashes, always explains --------------------------

@pytest.mark.parametrize("bad", [
    pytest.param("", id="empty"),
    pytest.param("nonsense", id="no-dots"),
    pytest.param("A.B.C", id="wrong-scheme"),
    pytest.param("AUDITBOX-LIC-1.!!!.!!!", id="undecodable"),
    pytest.param(None, id="None"),
])
def test_malformed_input_raises_licence_error_and_nothing_else(pub, bad):
    with pytest.raises(LicenceError):
        verify(pub, bad)


# -- refuses to issue something meaningless ----------------------------------

@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(customer="", expires=FUTURE, frameworks=["PQC"]), id="no-customer"),
    pytest.param(dict(customer="X", expires=FUTURE, frameworks=[]), id="no-frameworks"),
])
def test_refuses_to_issue_something_meaningless(priv, kwargs):
    with pytest.raises(LicenceError):
        issue(priv, **kwargs)


# -- the signing key is never read from the repo -----------------------------

def test_an_absent_signing_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("AUDITBOX_LICENCE_KEY", raising=False)
    with pytest.raises(LicenceError) as e:
        private_key_from_env()
    assert "environment" in str(e.value).lower()


def test_the_signing_key_is_read_from_the_environment(monkeypatch, keypair):
    monkeypatch.setenv("AUDITBOX_LICENCE_KEY", keypair["priv_pem"].decode())
    key = issue(private_key_from_env(), customer="Y", expires=FUTURE,
                frameworks=["ISO27001"])
    assert key.startswith("AUDITBOX-LIC-1.")


# -- the public key cannot mint licences -------------------------------------

def test_the_public_key_cannot_sign(pub):
    assert not hasattr(pub, "sign")


def test_the_keypair_round_trips_through_pem(keypair):
    assert load_public_key(keypair["pub_pem"]).public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ) == keypair["pub_pem"]


# -- determinism -------------------------------------------------------------

def test_identical_inputs_produce_an_identical_key(priv):
    """Reissuing the same licence must not produce a different file to ship."""
    args = dict(customer="Z", expires=FUTURE, frameworks=["PQC"], licence_id="fixed",
                issued=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc))
    assert issue(priv, **args) == issue(priv, **args)
