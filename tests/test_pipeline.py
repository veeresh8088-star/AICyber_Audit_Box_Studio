"""Gates must actually stop the build. A gate that warns is not a gate.

Every step is exercised with an injected fake, so the chain is tested without
Docker, without a model, and without a network -- and so a failing step can be
proven to prevent the artifact rather than merely being reported.
"""
import subprocess
from datetime import date, timedelta

import pytest

from studio.config import Profile
from studio import pipeline as pl


def make_profile(**over):
    base = dict(
        licence=dict(customer="STPI",
                     expires=(date.today() + timedelta(days=365)).isoformat(),
                     frameworks=["PQC"]),
        hardware=dict(physical_cores=32, ram_gb=125),
        model="google_gemma-4-E4B-it-Q4_K_M.gguf",
    )
    base.update(over)
    return Profile(**base)


@pytest.fixture
def prof():
    return make_profile()


FINDINGS = [{"package": "pillow", "version": "12.2.0", "severity": "HIGH"},
            {"package": "setuptools", "version": "81.0.0", "severity": "MODERATE"}]


# -- SCA gate stops the build at the configured severity ---------------------

def test_a_high_finding_fails_the_build(prof):
    r = pl.step_sca(prof, lambda: FINDINGS)
    assert not r.ok, r.detail
    assert "pillow" in r.detail, r.detail


def test_the_threshold_is_respected(prof):
    r = pl.step_sca(make_profile(build=dict(fail_on_sca_severity="CRITICAL")),
                    lambda: FINDINGS)
    assert r.ok, r.detail
    r = pl.step_sca(prof, lambda: [{"package": "x", "version": "1", "severity": "LOW"}])
    assert r.ok, r.detail


def test_a_disabled_scan_is_marked_skipped_not_passed_silently():
    r = pl.step_sca(make_profile(build=dict(run_sca=False)), lambda: FINDINGS)
    assert r.ok and r.skipped


# -- test gate ---------------------------------------------------------------

def test_all_green_passes(prof):
    assert pl.step_tests(prof, lambda: (130, 0, "")).ok


def test_any_failing_test_stops_the_build(prof):
    r = pl.step_tests(prof, lambda: (128, 2, "test_zero_controls"))
    assert not r.ok, r.detail
    assert "test_zero_controls" in r.detail, r.detail


# -- compile gate ------------------------------------------------------------

def test_skipping_compilation_is_flagged_loudly():
    r = pl.step_compile(make_profile(build=dict(compile_source=False)), lambda: (True, ""))
    assert r.skipped and "SOURCE WILL SHIP" in r.detail, r.detail


def test_a_failed_compile_stops_the_build(prof):
    r = pl.step_compile(prof, lambda: (False, "nuitka failed on langgraph dynamic import"))
    assert not r.ok, r.detail


# -- shape resolution refuses an illegal patch -------------------------------

@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """v1, v2 (code only), v3 (dependency bump)."""
    path = tmp_path_factory.mktemp("piperepo")

    def git(*a):
        return subprocess.run(["git", "-C", str(path), *a],
                              capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (path / "src").mkdir()
    (path / "src" / "a.py").write_text("1")
    (path / "requirements.txt").write_text("a==1\n")
    git("add", "-A"); git("commit", "-qm", "1"); git("tag", "v1")
    (path / "src" / "a.py").write_text("2")
    git("add", "-A"); git("commit", "-qm", "2"); git("tag", "v2")
    (path / "requirements.txt").write_text("a==2\n")
    git("add", "-A"); git("commit", "-qm", "3"); git("tag", "v3")
    return str(path)


def test_a_legal_patch_is_accepted(repo):
    r = pl.step_resolve_shape(make_profile(bundle="patch", patch_from="v1"),
                              repo, "v2", None)
    assert r.ok and r.data.get("shape") == "patch", r.detail


def test_a_patch_over_a_dependency_bump_is_refused(repo):
    r = pl.step_resolve_shape(make_profile(bundle="patch", patch_from="v2"),
                              repo, "v3", None)
    assert not r.ok, r.detail
    assert "requirements.txt" in r.detail, r.detail


def test_auto_downgrades_to_full_when_needed(repo):
    r = pl.step_resolve_shape(make_profile(), repo, "v3", "v2")
    assert r.ok and r.data["shape"] == "full", r.detail


def test_auto_chooses_patch_when_safe(repo):
    r = pl.step_resolve_shape(make_profile(), repo, "v2", "v1")
    assert r.ok and r.data["shape"] == "patch", r.detail


def test_an_explicit_full_never_diffs(repo):
    r = pl.step_resolve_shape(make_profile(bundle="full"), repo, "v3", "v1")
    assert r.ok and r.data["shape"] == "full", r.detail


def test_no_previous_version_means_full(repo):
    r = pl.step_resolve_shape(make_profile(), repo, "v2", None)
    assert r.ok and r.data["shape"] == "full", r.detail


# -- a step that raises does not take the chain down -------------------------

def test_an_exception_becomes_a_failed_step():
    def boom():
        raise RuntimeError("docker daemon unreachable")
    r = pl._timed(boom)
    assert not r.ok and "docker daemon" in r.detail, r.detail


# -- the report refuses to look successful when a gate failed ----------------

def test_the_report_is_honest_about_a_failed_gate():
    rep = pl.BuildReport(profile="stpi", version="3.24", started="now")
    rep.steps = [pl.StepResult("tests", True, "130 passed"),
                 pl.StepResult("sca", False, "1 HIGH: pillow==12.2.0")]
    assert not rep.ok
    assert rep.failed_step.name == "sca"
    assert "no artifact" in rep.summary()
    assert "FAIL" in rep.summary()


# -- publish is honest when nothing is configured ----------------------------

def test_no_artifactory_is_skipped_not_published(prof):
    r = pl.step_publish(prof, "/tmp/x.tar", "3.24", None)
    assert r.skipped and "left locally" in r.detail, r.detail


def test_a_rejected_upload_fails_the_build(prof):
    r = pl.step_publish(prof, "/tmp/x.tar", "3.24", lambda a, v: (False, "401 unauthorized"))
    assert not r.ok, r.detail


# -- sizing step reports the real numbers ------------------------------------

def test_the_sizing_step_reports_the_real_numbers(prof):
    def fake_sizer(**kw):
        return {"np_slots": 32, "shared_pool": 1048576,
                "max_concurrent_audits": 16, "limited_by": "cores"}
    r = pl.step_sizing(prof, fake_sizer)
    assert "-np 32" in r.detail and "1,048,576" in r.detail, r.detail
