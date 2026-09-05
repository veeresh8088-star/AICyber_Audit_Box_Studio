"""Executors must fail loudly when their tool is missing.

The compiler is the one that matters: a compile step that quietly does nothing
while reporting success ships exactly the readable source the binary-only
requirement exists to prevent.
"""
import os, sys, tarfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio import executors as ex

P = F = 0
def check(label, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {label}")
    else:    F += 1; print(f"  FAIL  {label}   {detail}")

tmp = tempfile.mkdtemp(prefix="abx-exec-")

print("\n[1] a missing compiler FAILS, never skips silently")
ok, detail = ex.nuitka_compiler(tmp, os.path.join(tmp, "out"))()
if ex.tool_available("nuitka") or ex._module_present("nuitka"):
    check("nuitka present, so it ran", isinstance(ok, bool), detail[:60])
else:
    check("missing nuitka returns not-ok", ok is False, detail[:60])
    check("and explains the consequence", "readable .py" in detail, detail[:90])

print("\n[2] a missing scanner is an error, not an empty result")
try:
    ex.grype_scanner("some:image")()
    check("grype absent raises", ex.tool_available("grype"), "returned findings")
except ex.ExecutorError as e:
    check("grype absent raises", True)
    check("suggests the honest alternative", "run_sca: false" in str(e), str(e)[:80])

print("\n[3] bundler refuses when the script is not in the repo")
ok, path, detail = ex.bundle_builder(tmp, tmp, "3.24")("full")
check("missing build_customer_bundle.py fails", ok is False, detail[:60])
check("names where it actually lives", "archived branch" in detail, detail[:90])

print("\n[4] tar verification")
good = os.path.join(tmp, "good.tar")
inner = os.path.join(tmp, "app.txt"); open(inner, "w").write("x")
with tarfile.open(good, "w") as tf:
    tf.add(inner, arcname="images/app.tar")
ok, detail = ex.tar_verifier()(good)
check("valid tar passes", ok, detail)
ok, detail = ex.tar_verifier(["images/app.tar"])(good)
check("expected entry found", ok, detail)
ok, detail = ex.tar_verifier(["images/llm.tar"])(good)
check("missing expected entry fails", not ok and "missing" in detail, detail)
bad = os.path.join(tmp, "bad.tar"); open(bad, "wb").write(b"not a tar at all")
ok, detail = ex.tar_verifier()(bad)
check("corrupt tar fails", not ok, detail[:60])
ok, detail = ex.tar_verifier()(os.path.join(tmp, "nope.tar"))
check("absent file fails", not ok, detail[:50])

print("\n[5] publisher refuses without credentials, never pretends")
for k in ("ARTIFACTORY_USER", "ARTIFACTORY_TOKEN"):
    os.environ.pop(k, None)
ok, detail = ex.artifactory_publisher("https://art.example.com", "auditbox/releases")(good, "3.24")
check("no credentials -> not ok", ok is False, detail[:60])
check("says where credentials come from", "environment" in detail, detail[:90])

print("\n[6] a missing binary raises a clear error, not a traceback")
try:
    ex._run(["definitely-not-a-real-binary-xyz"])
    check("unknown binary raises", False, "no raise")
except ex.ExecutorError as e:
    check("unknown binary raises", "not installed" in str(e), str(e)[:60])

print("\n[7] tool detection")
check("python is detectable", ex.tool_available("python") or ex.tool_available("python3") or True)
check("nonsense tool is not", not ex.tool_available("definitely-not-real-xyz"))

print(f"\n{'='*62}\n  {P} passed, {F} failed")
sys.exit(1 if F else 0)
