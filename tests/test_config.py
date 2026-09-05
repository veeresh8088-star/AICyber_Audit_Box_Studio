"""A profile must refuse configurations that cannot work, at build time.

The point of validating here is that the failure lands on the person cutting
the release, who can fix it, rather than on the customer three weeks later.
"""
import os, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from pydantic import ValidationError
from studio.config import Profile, Framework, ModelChoice, BundleShape

P = F = 0
def check(label, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {label}")
    else:    F += 1; print(f"  FAIL  {label}   {detail}")

FUTURE = (date.today() + timedelta(days=200)).isoformat()
def prof(**over):
    base = dict(
        licence=dict(customer="ACME", expires=FUTURE, frameworks=["PQC"]),
        hardware=dict(physical_cores=32, ram_gb=125),
        model="google_gemma-4-E4B-it-Q4_K_M.gguf",
    )
    base.update(over)
    return Profile(**base)

def rejects(why, **over):
    try:
        prof(**over); check(f"rejects {why}", False, "accepted")
    except ValidationError as e:
        check(f"rejects {why}", True)

print("\n[1] the real profiles on disk parse")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("stpi.yaml", "smallsite.yaml"):
    path = os.path.join(here, "profiles", name)
    p = Profile(**yaml.safe_load(open(path, encoding="utf-8")))
    check(f"{name} valid", p.licence.customer != "", p.licence.customer)
stpi = Profile(**yaml.safe_load(open(os.path.join(here,"profiles","stpi.yaml"), encoding="utf-8")))
check("STPI is PQC only", stpi.licence.frameworks == [Framework.PQC], str(stpi.licence.frameworks))

print("\n[2] a model that cannot fit is refused before it ships")
rejects("12B on a 16GB box", model="gemma-4-12B-it-Q8_0.gguf", hardware=dict(physical_cores=8, ram_gb=16))
ok = prof(model="gemma-4-12B-it-Q8_0.gguf", hardware=dict(physical_cores=8, ram_gb=32))
check("12B on 32GB is allowed", ok.model == ModelChoice.GEMMA4_12B_Q8)
ok = prof(model="google_gemma-4-E4B-it-Q4_K_M.gguf", hardware=dict(physical_cores=4, ram_gb=16))
check("E4B on 16GB is allowed", ok.hardware.ram_gb == 16)

print("\n[3] a patch needs a base, and a base needs a patch")
rejects("patch without patch_from", bundle="patch")
rejects("patch_from without patch", bundle="full", patch_from="3.22")
ok = prof(bundle="patch", patch_from="3.22")
check("patch with a base is valid", ok.patch_from == "3.22")

print("\n[4] a licence must grant something")
rejects("no frameworks", licence=dict(customer="X", expires=FUTURE, frameworks=[]))
rejects("duplicate frameworks", licence=dict(customer="X", expires=FUTURE, frameworks=["PQC","PQC"]))
rejects("unknown framework", licence=dict(customer="X", expires=FUTURE, frameworks=["NOTREAL"]))
rejects("empty customer", licence=dict(customer="", expires=FUTURE, frameworks=["PQC"]))

print("\n[5] hardware must be plausible")
rejects("zero cores", hardware=dict(physical_cores=0, ram_gb=64))
rejects("zero RAM", hardware=dict(physical_cores=8, ram_gb=0))
rejects("context not a multiple of 1024", hardware=dict(physical_cores=8, ram_gb=64, ctx_per_request=30000))
ok = prof(hardware=dict(physical_cores=8, ram_gb=64, ctx_per_request=65536))
check("65536 context accepted", ok.hardware.ctx_per_request == 65536)

print("\n[6] frameworks can never be a customer-adjustable setting")
rejects("locking an unknown setting", runtime=dict(locked=["frameworks"]))
ok = prof(runtime=dict(locked=["max_concurrent_audits"]))
check("locking a real setting is fine", ok.runtime.locked == ["max_concurrent_audits"])

print("\n[7] build gates are validated")
rejects("nonsense severity", build=dict(fail_on_sca_severity="SPICY"))
ok = prof(build=dict(fail_on_sca_severity="critical"))
check("severity is normalised to upper case", ok.build.fail_on_sca_severity == "CRITICAL")
ok = prof(build=dict(compile_source=False))
check("compilation can be turned off explicitly", ok.build.compile_source is False)

print("\n[8] defaults are the safe ones")
d = prof()
check("compiles by default", d.build.compile_source is True)
check("encrypts by default", d.build.encrypt_bundle is True)
check("tests by default", d.build.run_tests is True)
check("scans by default", d.build.run_sca is True)
check("blocks on HIGH by default", d.build.fail_on_sca_severity == "HIGH")
check("bundle shape defaults to auto", d.bundle == BundleShape.AUTO)

print(f"\n{'='*62}\n  {P} passed, {F} failed")
sys.exit(1 if F else 0)
