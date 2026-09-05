"""Licence keys must resist a customer editing their own entitlements.

The product's existing licence is a token wallet with an expiry -- no field
says which frameworks were bought, so a PQC-only customer cannot be served.
These keys carry that, signed, so the answer cannot be changed by the party it
constrains.
"""
import base64, json, os, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio.licensing import (
    generate_keypair, load_private_key, load_public_key, issue, verify,
    LicenceError, private_key_from_env,
)

P = F = 0
def check(label, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {label}")
    else:    F += 1; print(f"  FAIL  {label}   {detail}")

priv_pem, pub_pem = generate_keypair()
priv, pub = load_private_key(priv_pem), load_public_key(pub_pem)
future = date.today() + timedelta(days=365)

print("\n[1] a PQC-only licence grants PQC and nothing else")
key = issue(priv, customer="STPI", expires=future, frameworks=["PQC"])
lic = verify(pub, key)
check("customer preserved", lic.customer == "STPI", lic.customer)
check("permits PQC", lic.permits("PQC"))
check("denies ISO27001", not lic.permits("ISO27001"))
check("denies VAPT", not lic.permits("VAPT"))
check("case-insensitive, denies 'pqc ' variants correctly", lic.permits("pqc") and lic.permits(" PQC "))
check("expiry preserved", lic.expires == future, str(lic.expires))
check("has a unique id", len(lic.licence_id) >= 8, lic.licence_id)

print("\n[2] tampering is detected")
head, body_b64, sig_b64 = key.split(".")
pad = "=" * (-len(body_b64) % 4)
payload = json.loads(base64.urlsafe_b64decode(body_b64 + pad))
payload["frameworks"] = ["PQC", "ISO27001", "VAPT"]          # customer grants themselves everything
forged_body = base64.urlsafe_b64encode(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
forged = f"{head}.{forged_body}.{sig_b64}"
try:
    verify(pub, forged); check("edited entitlements rejected", False, "forgery accepted")
except LicenceError as e:
    check("edited entitlements rejected", "signature" in str(e).lower(), str(e)[:60])

payload2 = json.loads(base64.urlsafe_b64decode(body_b64 + pad))
payload2["expires"] = (date.today() + timedelta(days=99999)).isoformat()
b2 = base64.urlsafe_b64encode(json.dumps(payload2, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
try:
    verify(pub, f"{head}.{b2}.{sig_b64}"); check("extended expiry rejected", False, "forgery accepted")
except LicenceError: check("extended expiry rejected", True)

print("\n[3] a different vendor's key cannot sign for us")
other_priv, _ = generate_keypair()
other = issue(load_private_key(other_priv), customer="STPI", expires=future, frameworks=["PQC"])
try:
    verify(pub, other); check("foreign signature rejected", False, "accepted")
except LicenceError: check("foreign signature rejected", True)

print("\n[4] expiry is enforced, and distinguishable from forgery")
past = issue(priv, customer="X", expires=date.today() - timedelta(days=1), frameworks=["PQC"])
try:
    verify(pub, past); check("expired licence rejected", False, "accepted")
except LicenceError as e:
    check("expired licence rejected", "expired" in str(e).lower(), str(e)[:70])
lic2 = verify(pub, past, check_expiry=False)
check("still decodable for a clear error message", lic2.customer == "X")
check("reports how long ago it lapsed", lic2.expired and lic2.days_remaining < 0, str(lic2.days_remaining))

print("\n[5] malformed input never crashes, always explains")
for bad, why in [("", "empty"), ("nonsense", "no dots"), ("A.B.C", "wrong scheme"),
                 ("AUDITBOX-LIC-1.!!!.!!!", "undecodable"), (None, "None")]:
    try:
        verify(pub, bad); check(f"rejects {why}", False, "accepted")
    except LicenceError: check(f"rejects {why}", True)
    except Exception as e: check(f"rejects {why}", False, f"wrong exception {type(e).__name__}")

print("\n[6] refuses to issue something meaningless")
for kwargs, why in [
    (dict(customer="", expires=future, frameworks=["PQC"]), "no customer"),
    (dict(customer="X", expires=future, frameworks=[]), "no frameworks"),
]:
    try:
        issue(priv, **kwargs); check(f"refuses {why}", False, "issued anyway")
    except LicenceError: check(f"refuses {why}", True)

print("\n[7] the signing key is never read from the repo")
os.environ.pop("AUDITBOX_LICENCE_KEY", None)
try:
    private_key_from_env(); check("absent key raises a clear error", False, "no raise")
except LicenceError as e:
    check("absent key raises a clear error", "environment" in str(e).lower(), str(e)[:60])
os.environ["AUDITBOX_LICENCE_KEY"] = priv_pem.decode()
check("reads the key from the environment when set",
      issue(private_key_from_env(), customer="Y", expires=future, frameworks=["ISO27001"]).startswith("AUDITBOX-LIC-1."))
os.environ.pop("AUDITBOX_LICENCE_KEY", None)

print("\n[8] the public key cannot mint licences")
check("public key has no sign method", not hasattr(pub, "sign"))
check("keypair round-trips through PEM", load_public_key(pub_pem).public_bytes(
        encoding=__import__("cryptography.hazmat.primitives.serialization", fromlist=["x"]).Encoding.PEM,
        format=__import__("cryptography.hazmat.primitives.serialization", fromlist=["x"]).PublicFormat.SubjectPublicKeyInfo,
      ) == pub_pem)

print("\n[9] determinism: same inputs, same signature")
a = issue(priv, customer="Z", expires=future, frameworks=["PQC"], licence_id="fixed", issued=__import__("datetime").datetime(2026,1,1,tzinfo=__import__("datetime").timezone.utc))
b = issue(priv, customer="Z", expires=future, frameworks=["PQC"], licence_id="fixed", issued=__import__("datetime").datetime(2026,1,1,tzinfo=__import__("datetime").timezone.utc))
check("identical inputs produce an identical key", a == b)

print(f"\n{'='*62}\n  {P} passed, {F} failed")
sys.exit(1 if F else 0)
