"""Gates must actually stop the build. A gate that warns is not a gate.

Every step is exercised with an injected fake, so the chain is tested without
Docker, without a model, and without a network -- and so a failing step can be
proven to prevent the artifact rather than merely being reported.
"""
import os, sys, subprocess, tempfile
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio.config import Profile, BundleShape
from studio import pipeline as pl

P = F = 0
def check(label, cond, detail=""):
    global P, F
    if cond: P += 1; print(f"  PASS  {label}")
    else:    F += 1; print(f"  FAIL  {label}   {detail}")

def make_profile(**over):
    base = dict(
        licence=dict(customer="STPI", expires=(date.today()+timedelta(days=365)).isoformat(),
                     frameworks=["PQC"]),
        hardware=dict(physical_cores=32, ram_gb=125),
        model="google_gemma-4-E4B-it-Q4_K_M.gguf",
    )
    base.update(over)
    return Profile(**base)

print("\n[1] SCA gate stops the build at the configured severity")
prof = make_profile()
findings = [{"package":"pillow","version":"12.2.0","severity":"HIGH"},
            {"package":"setuptools","version":"81.0.0","severity":"MODERATE"}]
r = pl.step_sca(prof, lambda: findings)
check("HIGH finding fails the build", not r.ok, r.detail[:60])
check("names the package", "pillow" in r.detail, r.detail[:60])
r = pl.step_sca(make_profile(build=dict(fail_on_sca_severity="CRITICAL")), lambda: findings)
check("threshold CRITICAL lets HIGH through", r.ok, r.detail[:60])
r = pl.step_sca(prof, lambda: [{"package":"x","version":"1","severity":"LOW"}])
check("only LOW findings pass", r.ok, r.detail[:60])
r = pl.step_sca(make_profile(build=dict(run_sca=False)), lambda: findings)
check("disabled scan is marked skipped, not passed silently", r.ok and r.skipped)

print("\n[2] test gate")
r = pl.step_tests(prof, lambda: (130, 0, ""))
check("all green passes", r.ok, r.detail)
r = pl.step_tests(prof, lambda: (128, 2, "test_zero_controls"))
check("any failure stops the build", not r.ok, r.detail[:60])
check("names what failed", "test_zero_controls" in r.detail, r.detail[:60])

print("\n[3] compile gate warns loudly when disabled")
r = pl.step_compile(make_profile(build=dict(compile_source=False)), lambda: (True, ""))
check("skipping compilation is flagged", r.skipped and "SOURCE WILL SHIP" in r.detail, r.detail)
r = pl.step_compile(prof, lambda: (False, "nuitka failed on langgraph dynamic import"))
check("a failed compile stops the build", not r.ok, r.detail[:60])

print("\n[4] shape resolution refuses an illegal patch")
tmp = tempfile.mkdtemp(prefix="abx-pipe-")
repo = os.path.join(tmp, "repo"); os.makedirs(repo)
def git(*a): return subprocess.run(["git","-C",repo,*a], capture_output=True, text=True)
git("init","-q"); git("config","user.email","t@t"); git("config","user.name","t")
os.makedirs(os.path.join(repo,"src"))
open(os.path.join(repo,"src","a.py"),"w").write("1")
open(os.path.join(repo,"requirements.txt"),"w").write("a==1\n")
git("add","-A"); git("commit","-qm","1"); git("tag","v1")
open(os.path.join(repo,"src","a.py"),"w").write("2")
git("add","-A"); git("commit","-qm","2"); git("tag","v2")
open(os.path.join(repo,"requirements.txt"),"w").write("a==2\n")
git("add","-A"); git("commit","-qm","3"); git("tag","v3")

p_patch = make_profile(bundle="patch", patch_from="v1")
r = pl.step_resolve_shape(p_patch, repo, "v2", None)
check("legal patch accepted", r.ok and r.data.get("shape")=="patch", r.detail[:60])
p_bad = make_profile(bundle="patch", patch_from="v2")
r = pl.step_resolve_shape(p_bad, repo, "v3", None)
check("patch over a dependency bump is REFUSED", not r.ok, r.detail[:80])
check("explains which file blocked it", "requirements.txt" in r.detail, r.detail[:80])
r = pl.step_resolve_shape(make_profile(), repo, "v3", "v2")
check("auto downgrades to full when needed", r.ok and r.data["shape"]=="full", r.detail[:60])
r = pl.step_resolve_shape(make_profile(), repo, "v2", "v1")
check("auto chooses patch when safe", r.ok and r.data["shape"]=="patch", r.detail[:60])
r = pl.step_resolve_shape(make_profile(bundle="full"), repo, "v3", "v1")
check("explicit full never diffs", r.ok and r.data["shape"]=="full", r.detail[:50])
r = pl.step_resolve_shape(make_profile(), repo, "v2", None)
check("no previous version -> full", r.ok and r.data["shape"]=="full", r.detail[:50])

print("\n[5] a step that raises does not take the chain down")
r = pl._timed(lambda: (_ for _ in ()).throw(RuntimeError("docker daemon unreachable")))
check("exception becomes a failed step", not r.ok and "docker daemon" in r.detail, r.detail[:60])

print("\n[6] the report refuses to look successful when a gate failed")
rep = pl.BuildReport(profile="stpi", version="3.24", started="now")
rep.steps = [pl.StepResult("tests", True, "130 passed"),
             pl.StepResult("sca", False, "1 HIGH: pillow==12.2.0")]
check("report is not ok", not rep.ok)
check("names the failing step", rep.failed_step.name == "sca")
check("summary says no artifact was produced", "no artifact" in rep.summary())
check("summary marks the failure", "FAIL" in rep.summary())

print("\n[7] publish is honest when nothing is configured")
r = pl.step_publish(prof, "/tmp/x.tar", "3.24", None)
check("no Artifactory -> skipped, not 'published'", r.skipped and "left locally" in r.detail, r.detail)
r = pl.step_publish(prof, "/tmp/x.tar", "3.24", lambda a, v: (False, "401 unauthorized"))
check("a rejected upload fails the build", not r.ok, r.detail)

print("\n[8] sizing step reports the real numbers")
def fake_sizer(**kw):
    return {"np_slots":32,"shared_pool":1048576,"max_concurrent_audits":16,"limited_by":"cores"}
r = pl.step_sizing(prof, fake_sizer)
check("reports -np and pool", "-np 32" in r.detail and "1,048,576" in r.detail, r.detail)

print(f"\n{'='*62}\n  {P} passed, {F} failed")
sys.exit(1 if F else 0)
