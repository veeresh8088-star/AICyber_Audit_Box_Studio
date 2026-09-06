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
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Callable, List, Optional, Tuple


class ExecutorError(Exception):
    pass


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 3600,
         tail: int = 4000) -> Tuple[int, str]:
    """Run a command, returning (exit code, combined output tail).

    tail is how much of the end to keep. 4000 characters is plenty for reading
    an error, and was not enough to find pytest's summary: the product prints
    database replication notices after the tests finish, and they pushed
    "258 passed, 6 skipped" out of the window entirely. Callers that parse the
    output rather than merely quote it ask for more.
    """
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise ExecutorError(f"{cmd[0]} is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise ExecutorError(f"{cmd[0]} timed out after {timeout}s")
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, out[-tail:]


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


# ── tests ────────────────────────────────────────────────────────────────────

# "258 passed, 6 skipped in 14.81s" -- a count, spaces, then the word. Written
# without escapes so it survives being edited through a shell.
_PYTEST_COUNT_RE = re.compile("([0-9]+) +(passed|failed|error|errors)")


def pytest_runner(repo: str, paths: Optional[List[str]] = None) -> Callable:
    """Run the product's test suite. Returns (passed, failed, detail).

    The summary line is matched rather than split on whitespace. Splitting
    produced the token "passed," -- with the comma pytest adds whenever anything
    else is reported -- which never equalled "passed", so a green run of 258
    tests was reported as "0 passed". That reads exactly like a run which
    collected nothing, and noticing that is the one thing this step is for.
    """
    def run():
        # No -q here. The product's pytest.ini already sets "addopts = -q", and a
        # second one makes -qq, which drops the summary line entirely -- so the
        # count this step exists to read was never printed. Verified against the
        # real suite: with -q the output ends at the progress dots; without it,
        # "258 passed, 6 skipped in 12.27s". Verbosity is the project's choice.
        cmd = [sys.executable, "-m", "pytest", *(paths or ["tests"])]
        # A generous tail: the summary is what this step reads, and anything the
        # suite prints after it competes for the same window.
        code, out = _run(cmd, cwd=repo, timeout=1800, tail=60000)
        passed = failed = 0
        summary = ""
        for line in reversed(out.splitlines()):
            counts = _PYTEST_COUNT_RE.findall(line)
            if not counts:
                continue
            for n, word in counts:
                if word == "passed":
                    passed = int(n)
                else:                     # failed, error and errors all stop the build
                    failed += int(n)
            summary = line.strip()
            break
        # The summary line, not the last line. This product prints database
        # replication notices after its tests, so the build report was quoting
        # "[REPLICATION SUCCESS] Synced Master -> Slave 1" as its test result.
        last = summary or (out.splitlines()[-1] if out else "exit %d" % code)
        if code != 0 and failed == 0:
            # A run that collected nothing must not pass: a gate that goes green
            # because no test executed is worse than no gate. Distinguished from
            # a genuine failure so whoever reads the report knows which it was.
            if "no tests ran" in out.lower() or "collected 0 items" in out.lower():
                return 0, 1, ("pytest collected no tests -- check the path. The "
                              "product's tests/ mixes pytest files with scripts "
                              "that sys.exit at import, which pytest cannot collect")
            failed = 1                    # non-zero exit with no parseable count
        if code == 0 and passed == 0:
            # Exited green and counted nothing: the summary was not understood,
            # which is how "0 passed" came to be reported as a pass.
            return 0, 1, ("pytest exited 0 but no test count could be read from its "
                          "output -- refusing to call that a pass. Last line: %s"
                          % last[:120])
        return passed, failed, last
    return run


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

def nuitka_compiler(repo: str, out_dir: str, packages: Optional[List[str]] = None,
                    image: str = "python:3.11-slim", timeout: int = 7200) -> Callable:
    """Compile src/ to native modules INSIDE the image the product ships on.

    Returns (ok, detail). This used to shell out to nuitka on the build machine,
    which was wrong in two ways at once. The product runs on python:3.11-slim,
    so a build on a Windows host produces a .pyd for the wrong platform and the
    wrong interpreter -- an artifact that could never go in the image. And when
    it was tried there anyway it segfaulted on import: compiled successfully,
    54 modules, then died in exec_module.

    Built inside the target image the same source produces
    src.cpython-311-x86_64-linux-gnu.so and imports cleanly, so the compiler was
    never the problem; the host was.

    A missing compiler is a FAILURE, never a silent skip: a compile step that
    does nothing while reporting success ships exactly the source the customer
    must not have.
    """
    def run():
        if not tool_available("docker"):
            return False, ("docker is not installed, and compilation has to happen "
                           "inside the image the product ships on. Install it, or set "
                           "compile_source: false to accept readable .py in the image.")
        targets = packages or ["src"]
        os.makedirs(out_dir, exist_ok=True)
        # Build, then IMPORT what was built. "It compiled" is not evidence it
        # works -- the Windows attempt compiled and segfaulted, and a bundle
        # whose modules cannot be imported is worse than one shipping source.
        script = (
            "set -e; "
            "apt-get update -qq >/dev/null 2>&1; "
            "apt-get install -y -qq gcc build-essential >/dev/null 2>&1; "
            "pip install -q nuitka; "
            "python -m nuitka --module %s --include-package=%s "
            "  --assume-yes-for-downloads --output-dir=/out; "
            "cd /out && python -c 'import %s' && echo NUITKA_IMPORT_OK"
            % (" ".join(targets), targets[0], targets[0])
        )
        code, out = _run(
            ["docker", "run", "--rm",
             "-v", "%s:/app:ro" % os.path.abspath(repo),
             "-v", "%s:/out" % os.path.abspath(out_dir),
             "-w", "/app", image, "sh", "-c", script],
            timeout=timeout)
        produced = [f for f in os.listdir(out_dir) if f.endswith((".so", ".pyd"))]
        if code != 0:
            return False, "nuitka failed in %s: %s" % (image, out[-300:])
        if not produced:
            return False, "nuitka reported success but produced no native modules"
        if "NUITKA_IMPORT_OK" not in out:
            return False, ("%s built but does not import -- refusing to ship a module "
                           "that fails at load. %s" % (produced[0], out[-200:]))
        return True, "%d native module(s) built in %s and imported" % (len(produced), image)
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
        # The bundler names files from this, so it gets the filename form.
        cmd = [sys.executable, script, "--version", artifact_version(version)]
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


# -- what a bundle must contain -----------------------------------------------

def licence_key_present(repo: str,
                        path: str = "config/licence_public.pem") -> Callable:
    """The verifying key has to be in the repo, or the bundle is unusable.

    Found by inspecting a real build: the file did not exist, was not tracked,
    and was absent from the app image -- .gitignore carried a blanket *.pem that
    kept it out. An installation with AUDITBOX_ENFORCE_ENTITLEMENTS=1 then finds
    no verifying key, fails closed, and refuses every framework. The customer
    gets a product that starts and audits nothing.

    Failing closed is the right behaviour for the runtime; noticing before the
    bundle ships is this step's job.
    """
    def run():
        full = os.path.join(repo, path)
        if not os.path.isfile(full):
            return False, (f"{path} is missing from {repo}. It ships with the product "
                           f"and verifies licences (it cannot mint them). Without it an "
                           f"installation with entitlement enforcement on refuses every "
                           f"framework. Generate one with 'studio keygen'.")
        body = open(full, "rb").read()
        if b"PUBLIC KEY" not in body:
            return False, f"{path} is not a public key -- refusing to ship it"
        if b"PRIVATE KEY" in body:
            # Shipping the signing key would let any customer mint their own licence.
            return False, f"{path} contains a PRIVATE key. This must never ship."
        return True, f"{path} present ({len(body)} bytes), public half only"
    return run


def artifact_version(version: str) -> str:
    """The version as it appears in a FILENAME, which is not how git spells it.

    Tags are v3.24; the artifacts this product has always produced are
    AICyberAuditBox-3.23-complete.tar and INSTALL_v3.23.md. Feeding the tag
    straight through produced "INSTALL_vv3.25.md" and a bundle directory that
    did not match the convention every previous release used. The tag keeps its
    prefix for git; anything that becomes a name loses it.
    """
    v = str(version).strip()
    return v[1:] if v[:1].lower() == "v" and v[1:2].isdigit() else v


def bundle_expectations(shape: str, version: str) -> List[str]:
    """The entries that must be inside the bundle tar, by shape.

    step_verify was being handed an empty expectation list, so it only ever
    proved the tar opened and was not empty. A bundle missing the images tar
    entirely -- the multi-gigabyte part, the only part that matters -- passed
    that check and would have been published.

    Names are matched as substrings, so the version-stamped prefix directory
    does not have to be reproduced exactly here.
    """
    version = artifact_version(version)
    if shape == "patch":
        return [
            f"AICyberAuditBox-{version}-patch",
            "apply_patch.sh",
            "apply_patch.bat",
            "Dockerfile.app.rebase",
            "src/",
        ]
    return [
        f"AICyberAuditBox-{version}",
        f"aicyberauditbox-images-{version}.tar",   # the images; without it nothing runs
        "docker-compose.yml",
        "install.sh",
        "install.bat",
        f"INSTALL_v{version}.md",
    ]


# -- are the weights actually in the image? -----------------------------------

DEFAULT_MODELS = (
    "gemma-4-12B-it-Q8_0.gguf",
    "google_gemma-4-E4B-it-Q4_K_M.gguf",
    "nomic-embed-text-v1.5.f16.gguf",
)


def image_model_verifier(llm_tag: str, expected: Optional[List[str]] = None,
                         min_bytes: int = 100 * 1024 * 1024) -> Callable:
    """Look inside the built LLM image and confirm the weights are there.

    Dockerfile.llm COPYs three .gguf into /models. verify_images_tar in the
    product's bundler confirms the image tag reached the tar; it cannot see
    into the layers. A cached layer, a renamed weight file or a COPY whose
    source was absent still yields an image that loads and starts -- and then
    fails on the customer's first inference, after the whole transfer.

    Runs against the local image on the build machine, where it was just built,
    so this costs one container start rather than a scan of several GB of tar.
    """
    def run():
        if not tool_available("docker"):
            raise ExecutorError(
                "docker is not installed or not on PATH, so the model weights "
                "inside the LLM image cannot be checked. Install it, or set "
                "verify_models: false in the profile and accept that a bundle "
                "can ship an image whose /models directory is empty."
            )
        want = list(expected or DEFAULT_MODELS)
        code, out = _run(["docker", "run", "--rm", "--entrypoint", "sh", llm_tag,
                          "-c", "ls -l /models 2>/dev/null || true"], timeout=300)
        if code != 0:
            return False, f"could not inspect {llm_tag}: {out[-200:]}"
        listing = out or ""
        missing, undersized = [], []
        for name in want:
            line = next((l for l in listing.splitlines() if name in l), None)
            if line is None:
                missing.append(name)
                continue
            size = next((int(tok) for tok in line.split() if tok.isdigit()
                         and int(tok) > 1024), 0)
            if size < min_bytes:
                undersized.append(f"{name} ({size:,} bytes)")
        if missing:
            return False, ("model weights absent from the image: "
                           + ", ".join(missing)
                           + " -- the image would start and fail at first inference")
        if undersized:
            # A weight file that exists but is tiny is the signature of a COPY
            # that picked up a Git LFS pointer rather than the real file.
            return False, "model weights are implausibly small: " + ", ".join(undersized)
        return True, f"{len(want)} model weight(s) present in {llm_tag}"
    return run


# -- integrity of the finished artifact ---------------------------------------

def checksum_writer() -> Callable:
    """Write a sha256 beside the artifact.

    A transfer that truncates a multi-gigabyte tar very often leaves something
    that still opens as a tar, so "it extracted" is not evidence it arrived
    whole. The customer checks this figure before spending an install on it.
    """
    def run(path: str):
        import hashlib
        if not os.path.isfile(path):
            return False, None, f"no artifact at {path}"
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        side = path + ".sha256"
        with open(side, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{digest}  {os.path.basename(path)}\n")
        return True, digest, f"sha256 {digest[:16]}... written to {os.path.basename(side)}"
    return run
