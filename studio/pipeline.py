"""The release chain, as gates.

Each step either passes or stops the build. Nothing is produced by a failed
run, because a bundle that exists is a bundle somebody will eventually ship:
the point of a gate is that the artifact never comes into being.

The steps mirror what was previously done by hand, in the order they have to
happen -- decide the shape before building, test before scanning, scan before
bundling, verify before encrypting, and only then publish.

Every step is a small object with a `run` that returns a StepResult, so the
chain can be executed dry (deciding and reporting without touching Docker) and
so each step is testable without the ones around it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

from studio.config import Profile, BundleShape
from studio.packaging import patch_is_legal, PatchDecision


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0
    data: dict = field(default_factory=dict)
    skipped: bool = False


@dataclass
class BuildReport:
    profile: str
    version: str
    started: str
    steps: List[StepResult] = field(default_factory=list)
    artifact: Optional[str] = None
    licence_key: Optional[str] = None
    sha256: Optional[str] = None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def failed_step(self) -> Optional[StepResult]:
        return next((s for s in self.steps if not s.ok), None)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d

    def summary(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'}  {self.profile} {self.version}"]
        for s in self.steps:
            mark = "skip" if s.skipped else ("ok" if s.ok else "FAIL")
            lines.append(f"   {mark:>4}  {s.name:<22} {s.detail[:70]}")
        if not self.ok:
            f = self.failed_step
            lines.append("")
            lines.append(f"   stopped at: {f.name}")
            lines.append("   no artifact was produced, and nothing was published.")
        return "\n".join(lines)


def _timed(fn: Callable[[], StepResult], name: str = "step") -> StepResult:
    """Run a step, timing it, converting any exception into a failed result.

    The name is passed in because an exception gives no clue which step raised,
    and "stopped at: unknown" tells whoever reads the build report nothing.
    """
    t0 = time.time()
    try:
        res = fn()
    except Exception as exc:                       # a step must never take the chain down
        return StepResult(name, False, f"{type(exc).__name__}: {exc}",
                          round(time.time() - t0, 1))
    res.seconds = round(time.time() - t0, 1)
    return res


# ── steps ────────────────────────────────────────────────────────────────────

def step_resolve_shape(profile: Profile, repo: str, version: str,
                       previous: Optional[str]) -> StepResult:
    """Decide full vs patch, and refuse an illegal patch."""
    if profile.bundle == BundleShape.FULL:
        return StepResult("resolve shape", True, "full bundle, requested explicitly",
                          data={"shape": "full"})
    base = profile.patch_from or previous
    if profile.bundle == BundleShape.PATCH:
        if not base:
            return StepResult("resolve shape", False,
                              "patch requested but no base version to patch from")
        d: PatchDecision = patch_is_legal(repo, base, version)
        if not d.legal:
            return StepResult("resolve shape", False,
                              f"patch from {base} is not valid: {d.reason}",
                              data={"blocking": d.blocking_changes})
        return StepResult("resolve shape", True, f"patch from {base}: {d.reason}",
                          data={"shape": "patch", "from": base})
    # auto
    if not base:
        return StepResult("resolve shape", True,
                          "no previous version known, building full",
                          data={"shape": "full"})
    d = patch_is_legal(repo, base, version)
    return StepResult("resolve shape", True,
                      f"auto -> {d.shape}: {d.reason}",
                      data={"shape": d.shape, "from": base if d.legal else None})


def step_working_tree(repo: str, lister) -> StepResult:
    """Refuse to build from a tree that does not match what git was asked about.

    The shape decision comes from a git diff; the images are built from the
    files on disk. A modified, uncommitted file is in the second and not the
    first, so it ships without ever being considered -- and if it is
    requirements.txt or a Dockerfile, it ships inside a patch that was declared
    legal precisely because git saw no such change.
    """
    dirty = lister(repo)
    if dirty:
        shown = ", ".join(dirty[:5]) + (" and %d more" % (len(dirty) - 5) if len(dirty) > 5 else "")
        return StepResult("working tree", False,
                          "uncommitted changes would be built but were not part of the "
                          "version comparison: " + shown +
                          ". Commit or stash them, then build.")
    return StepResult("working tree", True, "clean, so what git compared is what gets built")


def step_sizing(profile: Profile, sizer: Callable) -> StepResult:
    """Compute the customer's llama-server settings from their hardware."""
    s = sizer(
        physical_cores=profile.hardware.physical_cores,
        total_ram_gb=profile.hardware.ram_gb,
        model_gb=None,
        ctx_per_request=profile.hardware.ctx_per_request,
        max_audits_per_auditor=profile.runtime.max_audits_per_auditor,
    )
    return StepResult(
        "sizing", True,
        f"-np {s['np_slots']} -c {s['shared_pool']:,} - "
        f"{s['max_concurrent_audits']} concurrent audits - limited by {s['limited_by']}",
        data=s,
    )


def step_tests(profile: Profile, runner: Callable[[], tuple]) -> StepResult:
    """Run the product's own tests. A failure stops everything."""
    if not profile.build.run_tests:
        return StepResult("tests", True, "skipped by profile", skipped=True)
    passed, failed, detail = runner()
    if failed:
        return StepResult("tests", False, f"{failed} failing test(s) -- {detail}"[:200])
    return StepResult("tests", True, f"{passed} passed")


def step_sca(profile: Profile, scanner: Callable[[], list]) -> StepResult:
    """Dependency scan. Findings at or above the configured severity stop the build."""
    if not profile.build.run_sca:
        return StepResult("sca", True, "skipped by profile", skipped=True)
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    threshold = rank[profile.build.fail_on_sca_severity]
    findings = scanner()
    blocking = [f for f in findings
                if rank.get(str(f.get("severity", "")).upper(), 9) <= threshold]
    if blocking:
        names = ", ".join(sorted({f"{f['package']}=={f['version']}" for f in blocking})[:4])
        return StepResult("sca", False,
                          f"{len(blocking)} finding(s) at or above "
                          f"{profile.build.fail_on_sca_severity}: {names}",
                          data={"blocking": blocking})
    return StepResult("sca", True,
                      f"{len(findings)} advisory/advisories, none at or above "
                      f"{profile.build.fail_on_sca_severity}")


def step_compile(profile: Profile, compiler: Callable[[], tuple]) -> StepResult:
    """Compile src/ to native modules so no .py ships."""
    if not profile.build.compile_source:
        return StepResult("compile", True,
                          "skipped by profile -- SOURCE WILL SHIP IN THE IMAGE",
                          skipped=True)
    ok, detail = compiler()
    return StepResult("compile", ok, detail)


def step_bundle(profile: Profile, shape: str, builder: Callable[[str], tuple]) -> StepResult:
    ok, path, detail = builder(shape)
    return StepResult("bundle", ok, detail, data={"path": path} if ok else {})


def step_verify(bundle_path: str, verifier: Callable[[str], tuple]) -> StepResult:
    """Confirm the bundle contains what it should before anyone ships it."""
    ok, detail = verifier(bundle_path)
    return StepResult("verify", ok, detail)


def step_licence(profile: Profile, issuer: Callable[[Profile], str]) -> StepResult:
    key = issuer(profile)
    fw = ", ".join(f.value for f in profile.licence.frameworks)
    return StepResult("licence", True,
                      f"{profile.licence.customer}: {fw}, expires "
                      f"{profile.licence.expires.isoformat()}",
                      data={"licence_key": key})


def step_encrypt(profile: Profile, bundle_path: str, licence_key: str,
                 encryptor: Callable[[str, str], dict]) -> StepResult:
    if not profile.build.encrypt_bundle:
        return StepResult("encrypt", True, "skipped by profile", skipped=True)
    man = encryptor(bundle_path, licence_key)
    return StepResult("encrypt", True,
                      f"{man['encrypted_bytes']:,} bytes, "
                      f"sha256 {man['sha256_encrypted'][:16]}...", data=man)


def step_publish(profile: Profile, artifact: str, version: str,
                 publisher: Optional[Callable[[str, str], tuple]]) -> StepResult:
    """Push to Artifactory. Absent a publisher, say so rather than implying success."""
    if publisher is None:
        return StepResult("publish", True,
                          "no Artifactory configured -- artifact left locally",
                          skipped=True, data={"local": artifact})
    ok, detail = publisher(artifact, version)
    return StepResult("publish", ok, detail)


def step_verify_models(profile: Profile, verifier: Callable[[], tuple]) -> StepResult:
    """Confirm the weights are inside the LLM image, not merely that it exists.

    The product's bundler already proves every image TAG reached the images
    tar. It cannot see into the layers, so this is the gap it leaves: an image
    whose /models is empty saves, loads, starts, and fails at first inference,
    after the customer has taken the whole transfer.
    """
    if not profile.build.verify_models:
        return StepResult("verify models", True,
                          "skipped by profile -- THE IMAGE MAY SHIP WITHOUT WEIGHTS",
                          skipped=True)
    ok, detail = verifier()
    return StepResult("verify models", ok, detail)


def step_licence_key(checker: Callable[[], tuple]) -> StepResult:
    """Confirm the verifying key is in the repo before anything is built.

    Not optional and not profile-gated: a bundle without it is unusable at any
    setting, because the runtime cannot tell a real licence from a forged one
    and so trusts neither.
    """
    ok, detail = checker()
    return StepResult("licence key", ok, detail)


def step_checksum(profile: Profile, artifact: str,
                  writer: Callable[[str], tuple]) -> StepResult:
    """Write a sha256 beside the artifact for the customer to check on arrival.

    Not a gate on anything: it is the evidence that what arrived is what left.
    A truncated multi-gigabyte transfer commonly still opens as a tar.
    """
    if not profile.build.write_checksum:
        return StepResult("checksum", True, "skipped by profile", skipped=True)
    ok, digest, detail = writer(artifact)
    return StepResult("checksum", ok, detail, data={"sha256": digest} if ok else {})
