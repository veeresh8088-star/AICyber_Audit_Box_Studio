"""The release chain, in one place, as a generator of results.

Written because the same chain now has two front ends. A command line and a web
page that each build the sequence themselves would drift, and the failure mode
of that drift is specific and bad: one path grows a gate the other lacks, and
the artifact somebody actually ships is the one that skipped the check. So the
order of the steps and the conditions on them live here, once, and both callers
consume the same generator.

It yields StepResult as each step finishes rather than returning a list at the
end, because a build takes minutes and the caller needs to show progress. The
generator stops at the first failure -- a gate that only warns is not a gate,
and nothing after a failed one should run.
"""
from __future__ import annotations

import datetime
import os
from typing import Callable, Iterator, Optional

from studio import executors as ex
from studio import pipeline as pl
from studio.config import Profile
from studio.packaging import encrypt_bundle


class ChainContext:
    """What the chain produced, filled in as it goes.

    The caller reads this after the generator finishes: the CLI prints it, the
    web page renders it. Absent fields mean the chain stopped before that step.
    """

    def __init__(self, profile: Profile, version: str, out_dir: str):
        self.report = pl.BuildReport(
            profile=profile.licence.customer, version=version,
            started=datetime.datetime.now().isoformat(timespec="seconds"))
        self.out_dir = out_dir
        self.shape: Optional[str] = None
        self.bundle_path: Optional[str] = None
        self.artifact: Optional[str] = None

    @property
    def licence_key(self) -> Optional[str]:
        return self.report.licence_key

    @property
    def sha256(self) -> Optional[str]:
        return getattr(self.report, "sha256", None)


def run_chain(profile: Profile, ctx: ChainContext, *, repo: str, version: str,
              previous: Optional[str] = None, test_path=None,
              scan_target: Optional[str] = None,
              issuer: Callable[[Profile], str],
              sizer: Callable,
              publisher: Optional[Callable] = None) -> Iterator[pl.StepResult]:
    """Yield each step's result in order, stopping at the first failure."""
    out_dir = ctx.out_dir

    def record(step: pl.StepResult) -> pl.StepResult:
        ctx.report.steps.append(step)
        return step

    # 1 shape -- decided before anything is built, so an illegal patch costs
    # seconds rather than a multi-gigabyte build nobody can ship.
    step = record(pl._timed(
        lambda: pl.step_resolve_shape(profile, repo, version, previous), "resolve shape"))
    yield step
    if not step.ok:
        return
    ctx.shape = step.data.get("shape", "full")
    patch_from = step.data.get("from")

    # 1b the verifying key. Cheap, and a bundle without it is unusable at the
    # customer site whatever else is right, so it goes before the slow steps.
    step = record(pl._timed(
        lambda: pl.step_licence_key(ex.licence_key_present(repo)), "licence key"))
    yield step
    if not step.ok:
        return

    for name, make in (
        ("sizing", lambda: pl.step_sizing(profile, sizer)),
        ("tests", lambda: pl.step_tests(profile, ex.pytest_runner(repo, test_path))),
        # A missing compiler fails here rather than quietly shipping source.
        ("compile", lambda: pl.step_compile(
            profile, ex.nuitka_compiler(repo, os.path.join(out_dir, "compiled")))),
        ("sca", lambda: pl.step_sca(profile, ex.grype_scanner(scan_target or repo))),
    ):
        step = record(pl._timed(make, name))
        yield step
        if not step.ok:
            return

    # 6 bundle
    step = record(pl._timed(lambda: pl.step_bundle(
        profile, ctx.shape,
        ex.bundle_builder(repo, out_dir, version, patch_from)), "bundle"))
    yield step
    if not step.ok:
        return
    ctx.bundle_path = step.data["path"]

    # 7 what is in it, and 7b what is inside the image inside it. The second is
    # the only check that can see whether the model weights actually shipped.
    for name, make in (
        ("verify", lambda: pl.step_verify(
            ctx.bundle_path, ex.tar_verifier(ex.bundle_expectations(ctx.shape, version)))),
        ("verify models", lambda: pl.step_verify_models(
            profile, ex.image_model_verifier("aicyberauditbox-llm:%s" % version))),
    ):
        step = record(pl._timed(make, name))
        yield step
        if not step.ok:
            return

    # 8 licence
    step = record(pl._timed(lambda: pl.step_licence(profile, issuer), "licence"))
    yield step
    if not step.ok:
        return
    ctx.report.licence_key = step.data["licence_key"]

    # 9 encryption, keyed to that licence
    ctx.artifact = ctx.bundle_path
    if profile.build.encrypt_bundle:
        enc_path = ctx.bundle_path + ".enc"
        step = record(pl._timed(lambda: pl.step_encrypt(
            profile, ctx.bundle_path, ctx.report.licence_key,
            lambda src, key: encrypt_bundle(src, enc_path, key)), "encrypt"))
        yield step
        if not step.ok:
            return
        ctx.artifact = enc_path
    ctx.report.artifact = ctx.artifact

    # 9b the digest of whatever actually ships, encrypted or not
    step = record(pl._timed(
        lambda: pl.step_checksum(profile, ctx.artifact, ex.checksum_writer()), "checksum"))
    yield step
    if not step.ok:
        return
    ctx.report.sha256 = step.data.get("sha256")

    # 10 publish
    step = record(pl._timed(lambda: pl.step_publish(
        profile, ctx.artifact, version, publisher), "publish"))
    yield step
