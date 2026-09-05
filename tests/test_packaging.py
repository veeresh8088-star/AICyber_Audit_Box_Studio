"""Encryption must fail closed, and an illegal patch must be refused."""
import os, sys, tempfile, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio.packaging import (
    encrypt_bundle, decrypt_bundle, derive_key, patch_is_legal,
    PackagingError, LOWER_LAYER_PATHS,
)

P = F = 0
def check(label, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {label}")
    else:    F += 1; print(f"  FAIL  {label}   {detail}")

tmp = tempfile.mkdtemp(prefix="abx-pack-")
plain = os.path.join(tmp, "bundle.tar")
BODY = os.urandom(300_000)
open(plain, "wb").write(BODY)
LIC = "AUDITBOX-LIC-1.abc.def"

print("\n[1] round trip")
enc = os.path.join(tmp, "bundle.enc")
man = encrypt_bundle(plain, enc, LIC)
check("encrypted file written", os.path.getsize(enc) > 0)
check("ciphertext differs from plaintext", open(enc,"rb").read()[:64] != BODY[:64])
back = os.path.join(tmp, "back.tar")
res = decrypt_bundle(enc, back, LIC)
check("decrypts byte-identical", open(back,"rb").read() == BODY)
check("hash matches the manifest", res["sha256"] == man["sha256_plaintext"])

print("\n[2] the wrong licence cannot open it")
try:
    decrypt_bundle(enc, os.path.join(tmp,"x.tar"), "AUDITBOX-LIC-1.other.key")
    check("wrong licence rejected", False, "decrypted anyway")
except PackagingError as e:
    check("wrong licence rejected", "licence" in str(e).lower(), str(e)[:60])

print("\n[3] tampering is detected, not silently installed")
blob = bytearray(open(enc,"rb").read())
blob[-1] ^= 0xFF
bad = os.path.join(tmp,"tampered.enc"); open(bad,"wb").write(bytes(blob))
try:
    decrypt_bundle(bad, os.path.join(tmp,"y.tar"), LIC)
    check("modified bundle rejected", False, "accepted")
except PackagingError: check("modified bundle rejected", True)

print("\n[4] a foreign file fails clearly")
junk = os.path.join(tmp,"junk.bin"); open(junk,"wb").write(b"not a bundle at all")
try:
    decrypt_bundle(junk, os.path.join(tmp,"z.tar"), LIC)
    check("non-bundle rejected", False, "accepted")
except PackagingError as e:
    check("non-bundle rejected", "bundle" in str(e).lower(), str(e)[:60])

print("\n[5] key derivation")
s = os.urandom(16)
check("same licence+salt -> same key", derive_key(LIC, s) == derive_key(LIC, s))
check("different salt -> different key", derive_key(LIC, s) != derive_key(LIC, os.urandom(16)))
check("different licence -> different key", derive_key(LIC, s) != derive_key("other", s))
check("key is 32 bytes", len(derive_key(LIC, s)) == 32)
try:
    derive_key("", s); check("empty licence refused", False)
except PackagingError: check("empty licence refused", True)
check("salt is random per encryption",
      open(enc,"rb").read()[10:26] != encrypt_bundle(plain, os.path.join(tmp,"e2.enc"), LIC) and
      open(enc,"rb").read()[10:26] != open(os.path.join(tmp,"e2.enc"),"rb").read()[10:26])

print("\n[6] patch legality, against a real git repo")
repo = os.path.join(tmp, "repo"); os.makedirs(repo)
def git(*a): return subprocess.run(["git","-C",repo,*a], capture_output=True, text=True)
git("init","-q"); git("config","user.email","t@t"); git("config","user.name","t")
os.makedirs(os.path.join(repo,"src"))
open(os.path.join(repo,"src","app.py"),"w").write("v1")
open(os.path.join(repo,"requirements.txt"),"w").write("fastapi==1\n")
git("add","-A"); git("commit","-qm","v1"); git("tag","v1")

open(os.path.join(repo,"src","app.py"),"w").write("v2 code only")
git("add","-A"); git("commit","-qm","code change"); git("tag","v2")
d = patch_is_legal(repo, "v1", "v2")
check("code-only change -> patch legal", d.legal and d.shape == "patch", d.reason[:60])

open(os.path.join(repo,"requirements.txt"),"w").write("fastapi==2\n")
git("add","-A"); git("commit","-qm","dep bump"); git("tag","v3")
d = patch_is_legal(repo, "v2", "v3")
check("requirements change -> patch refused", not d.legal and d.shape == "full", d.reason[:70])
check("names the blocking file", "requirements.txt" in d.blocking_changes, str(d.blocking_changes))

d = patch_is_legal(repo, "v1", "v3")
check("spanning versions catches it too", not d.legal, d.reason[:60])
d = patch_is_legal(repo, "v2", "v2")
check("no change -> legal and says so", d.legal and "nothing changed" in d.reason, d.reason[:50])
d = patch_is_legal(repo, "v1", "v2", model_changed=True)
check("a changed model blocks a patch", not d.legal, str(d.blocking_changes))

print("\n[7] unknown refs fail loudly rather than defaulting to 'legal'")
try:
    patch_is_legal(repo, "v1", "does-not-exist")
    check("bad ref raises", False, "returned a decision")
except PackagingError: check("bad ref raises", True)

check("every lower-layer path is a real concern", all(
    p in ("requirements.txt","requirements.lock.txt","Dockerfile","Dockerfile.app",
          "Dockerfile.llm","docker-compose.yml","docker-compose.customer.yml")
    for p in LOWER_LAYER_PATHS))

print(f"\n{'='*62}\n  {P} passed, {F} failed")
sys.exit(1 if F else 0)
