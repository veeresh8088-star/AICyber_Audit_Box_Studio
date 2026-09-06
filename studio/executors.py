"""The steps that actually touch the outside world.

Kept apart from pipeline.py, which holds the gate logic, so the chain stays
testable without Docker, a compiler, a network or an Artifactory instance.
These are the implementations injected into it.

Every one of them fails loudly when its tool is missing. That matters most for
the compiler: a compile step that quietly does nothing produces a bundle full
of readable .py while reporting success, which is the exact failure the
binary-only requirement exists to prevent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from typing import Callable, List, Optional, Tuple


class ExecutorError(Exception):
    pass


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 3600) -> Tuple[int, str]:
    """Run a command, returning (exit code, combined output tail)."""
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise ExecutorError(f"{cmd[0]} is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise ExecutorError(f"{cmd[0]} timed out after {timeout}s")
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, out[-4000:]


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


# ── tests ────────────────────────────────────────────────────────────────────

def pytest_runner(repo: str, paths: Optional[List[str]] = None) -> Callable:
    """Run the product's test suite. Returns (passed, failed, detail)."""
    def run():
        cmd = [sys.executable, "-m", "pytest", "-q", *(paths or ["tests"])]
        code, out = _run(cmd, cwd=repo, timeout=1800)
        passed = failed = 0
        for line in reversed(out.splitlines()):
            if " passed" in line or " failed" in line:
                for part in line.replace(",", " ").split():
                    idx = line.split().index(part) if part in line.split() else -1
                tokens = line.split()
                for i, t in enumerate(tokens):
                    if t == "passed" and i: passed = int(tokens[i-1].strip("=") or 0)
                    if t == "failed" and i: failed = int(tokens[i-1].strip("=") or 0)
                break
        last = out.splitlines()[-1] if out else f"exit {code}"
        if code != 0 and failed == 0:
            # A run that collected nothing must not pass: a gate that goes green
            # because no test executed is worse than no gate. Distinguished from
            # a genuine failure so whoever reads the report knows which it was.
            if "no tests ran" in out.lower() or "collected 0 items" in out.lower():
                return 0, 1, ("pytest collected no tests -- check the path. The "
                              "product's tests/ mixes pytest files with scripts "
                              "that sys.exit at import, which pytest cannot collect")
            failed = 1                      # non-zero exit with no parseable count
        return passed, failed, last
    return run


# ── dependency scan ──────────────────────────────────────────────────────────

def grype_scanner(target: str) -> Callable:
    """Scan an image or directory with Grype. Returns a list of findings.

    Grype reads the built IMAGE, which catches the Debian packages inside
    python:3.11-slim that a Python lockfile cannot see.
    """
    def run():
        if not tool_available("grype"):
            raise ExecutorError(
                "grype is not installed. Install it, or set run_sca: false in the "
                "profile and accept that no dependency scan will happen."
            )
        code, out = _run(["grype", target, "-o", "json"], timeout=900)
        if code != 0:
            raise ExecutorError(f"grype failed: {out[-300:]}")
        try:
            data = json.loads(out[out.index("{"):])
        except (ValueError, json.JSONDecodeError) as exc:
            raise ExecutorError(f"grype output was not JSON: {exc}")
        findings = []
        for m in data.get("matches", []):
            art, vul = m.get("artifact", {}), m.get("vulnerability", {})
            findings.append({
                "package": art.get("name", "?"),
                "version": art.get("version", "?"),
                "severity": str(vul.get("severity", "UNKNOWN")).upper(),
                "id": vul.get("id", ""),
            })
        return findings
    return run


# ── compile ──────────────────────────────────────────────────────────────────

def nuitka_compiler(repo: str, out_dir: str, packages: Optional[List[str]] = None) -> Callable:
    """Compile src/ to native modules so no readable .py ships.

    Returns (ok, detail). Missing Nuitka is a FAILURE, never a silent skip: a
    compile step that does nothing while reporting success ships exactly the
    source the customer must not have.
    """
    def run():
        if not tool_available("nuitka") and not _module_present("nuitka"):
            return False, ("nuitka is not installed -- refusing to continue, because "
                           "skipping compilation would ship readable .py to the customer. "
                           "Install nuitka, or set compile_source: false to accept that.")
        targets = packages or ["src"]
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, "-m", "nuitka", "--module", "--assume-yes-for-downloads",
               f"--output-dir={out_dir}", *targets]
        code, out = _run(cmd, cwd=repo, timeout=7200)
        if code != 0:
            return False, f"nuitka failed: {out[-300:]}"
        produced = [f for f in os.listdir(out_dir) if f.endswith((".pyd", ".so"))]
        if not produced:
            return False, "nuitka reported success but produced no native modules"
        return True, f"{len(produced)} native module(s) built"
    return run


def _module_present(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── bundle ───────────────────────────────────────────────────────────────────

def bundle_builder(repo: str, out_dir: str, version: str,
                   patch_from: Optional[str] = None) -> Callable:
    """Call the product's own build_customer_bundle.py.

    Reused rather than reimplemented: it already produces the three shapes and
    already verifies its own tar, reading manifest.json to confirm every image
    tag made it in "rather than discovering it at the customer site after
    another multi-GB transfer".
    """
    def run(shape: str):
        script = os.path.join(repo, "build_customer_bundle.py")
        if not os.path.isfile(script):
            return False, None, (
                f"build_customer_bundle.py is not in {repo} -- point --repo at the "
                f"product repository, which is where that script lives."
            )
        cmd = [sys.executable, script, "--version", version]
        if shape == "full":
            cmd.append("--full")
        elif shape == "patch":
            if not patch_from:
                return False, None, "patch requested with no base version"
            cmd += ["--patch", patch_from]
        code, out = _run(cmd, cwd=repo, timeout=7200)
        if code != 0:
            return False, None, f"bundle failed: {out[-300:]}"
        produced = _newest_tar(out_dir) or _newest_tar(repo)
        if not produced:
            return False, None, "bundle reported success but no tar was found"
        return True, produced, f"{shape} bundle: {os.path.basename(produced)}"
    return run


def _newest_tar(directory: str) -> Optional[str]:
    try:
        tars = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tar")]
    except OSError:
        return None
    return max(tars, key=os.path.getmtime) if tars else None


def tar_verifier(expected_names: Optional[List[str]] = None) -> Callable:
    """Confirm a bundle is a readable tar containing what it should.

    Checked before publishing, because a corrupt bundle discovered at the
    customer site costs another multi-gigabyte transfer.
    """
    def run(path: str):
        if not os.path.isfile(path):
            return False, f"no bundle at {path}"
        try:
            with tarfile.open(path) as tf:
                names = tf.getnames()
        except tarfile.TarError as exc:
            return False, f"bundle is not a readable tar: {exc}"
        if not names:
            return False, "bundle is empty"
        missing = [n for n in (expected_names or []) if not any(n in x for x in names)]
        if missing:
            return False, f"bundle is missing: {', '.join(missing)}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return True, f"{len(names)} entries, {size_mb:,.0f} MB"
    return run


# ── publish ──────────────────────────────────────────────────────────────────

def artifactory_publisher(base_url: str, repo_path: str,
                          user_env: str = "ARTIFACTORY_USER",
                          token_env: str = "ARTIFACTORY_TOKEN") -> Callable:
    """Upload to Artifactory, tagged by version and date.

    Credentials come from the environment, never from a profile: a profile is
    version-controlled, and a token committed once stays in history.
    """
    def run(artifact: str, version: str):
        import datetime
        import urllib.request
        user = os.environ.get(user_env, "").strip()
        token = os.environ.get(token_env, "").strip()
        if not user or not token:
            return False, (f"{user_env}/{token_env} are not set -- cannot publish. "
                           f"Credentials are read from the environment, not the profile.")
        stamp = datetime.date.today().isoformat()
        name = f"{os.path.basename(artifact).rsplit('.',1)[0]}-{version}-{stamp}.tar"
        url = f"{base_url.rstrip('/')}/{repo_path.strip('/')}/{name}"
        try:
            with open(artifact, "rb") as fh:
                req = urllib.request.Request(url, data=fh.read(), method="PUT")
                auth = __import__("base64").b64encode(f"{user}:{token}".encode()).decode()
                req.add_header("Authorization", f"Basic {auth}")
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    if resp.status not in (200, 201):
                        return False, f"Artifactory returned {resp.status}"
        except Exception as exc:
            return False, f"upload failed: {exc}"
        return True, f"published as {name}"
    return run
