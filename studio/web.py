"""A local page for people who do not live in a terminal.

The chain was command-line only, which is fine for whoever wrote it and no use
to the person who was actually meant to cut releases. This serves one page on
localhost that lists the customer profiles, shows what a build would do, runs
it, and reports each gate as it passes or fails.

Built on http.server from the standard library on purpose. The studio has three
dependencies and every one of them has to be installable on a build machine that
may be offline; a web framework for a single-user local page would be a fourth
for no gain. For the same reason the page carries its own CSS and JS inline and
loads nothing from a CDN.

It binds to 127.0.0.1. There is no authentication, and it can start builds and
read licence keys, so it must not be exposed on a network.
"""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import yaml
from pydantic import ValidationError

from studio.chain import ChainContext, run_chain
from studio.config import Profile, BundleShape
from studio.licensing import issue, private_key_from_env, LicenceError
from studio.packaging import patch_is_legal, PackagingError
from studio.sizing import size_for_profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAGE = os.path.join(HERE, "static", "index.html")


# ── jobs ─────────────────────────────────────────────────────────────────────

class Job:
    """One build, running on its own thread so the page can poll it.

    A build takes minutes. Holding the HTTP request open for it would leave the
    operator with a spinner and no idea which gate is running, which is the
    thing the page exists to show.
    """

    def __init__(self, job_id: str):
        self.id = job_id
        self.steps: list = []
        self.done = False
        self.ok: Optional[bool] = None
        self.error: Optional[str] = None
        self.artifact: Optional[str] = None
        self.licence_key: Optional[str] = None
        self.sha256: Optional[str] = None
        self.shape: Optional[str] = None
        self._lock = threading.Lock()

    def add(self, step) -> None:
        with self._lock:
            self.steps.append({
                "name": step.name, "ok": bool(step.ok), "detail": step.detail,
                "seconds": step.seconds, "skipped": bool(step.skipped),
            })

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id, "done": self.done, "ok": self.ok,
                "error": self.error, "steps": list(self.steps),
                "artifact": self.artifact, "licence_key": self.licence_key,
                "sha256": self.sha256, "shape": self.shape,
            }


JOBS: Dict[str, Job] = {}


# ── helpers shared with the CLI ──────────────────────────────────────────────

def load_profile(path: str) -> Profile:
    with open(path, "r", encoding="utf-8") as fh:
        return Profile(**(yaml.safe_load(fh) or {}))


def profile_paths(directory: str) -> list:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.endswith((".yaml", ".yml")))


def profile_summary(path: str) -> dict:
    """Everything the page shows for one customer, or why it cannot be shown."""
    try:
        p = load_profile(path)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        return {"file": os.path.basename(path), "valid": False, "error": str(exc)[:400]}
    s = size_for_profile(p)
    return {
        "file": os.path.basename(path), "valid": True,
        "customer": p.licence.customer,
        "frameworks": [f.value for f in p.licence.frameworks],
        "expires": p.licence.expires.isoformat(),
        "seats": p.licence.seats,
        "model": p.model.value,
        "cores": p.hardware.physical_cores, "ram_gb": p.hardware.ram_gb,
        "bundle": p.bundle.value,
        "sizing": {
            "np_slots": s.np_slots, "shared_pool": s.shared_pool,
            "ctx_per_request": s.ctx_per_request,
            "max_concurrent_audits": s.max_concurrent_audits,
            "limited_by": s.limited_by, "projected_llm_gb": s.projected_llm_gb,
            "total_ram_gb": s.total_ram_gb, "headroom_gb": s.headroom_gb,
        },
        "gates": {
            "tests": p.build.run_tests, "sca": p.build.run_sca,
            "sca_severity": p.build.fail_on_sca_severity,
            "compile": p.build.compile_source, "encrypt": p.build.encrypt_bundle,
            "verify_models": p.build.verify_models,
            "write_checksum": p.build.write_checksum,
        },
    }


# ── hardware, edited in place ────────────────────────────────────────────────

_HW_FIELDS = ("physical_cores", "ram_gb", "ctx_per_request")


def _yaml_value(v) -> str:
    """Render a value the way the profiles already write it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def set_section(path: str, section: str, values: dict) -> dict:
    """Rewrite named fields inside one top-level block, leaving the rest alone.

    A yaml.safe_load/safe_dump round trip would reformat the profile and drop
    every comment in it -- including the ones recording why this customer is
    licensed for what they are, which is the reason these files are
    version-controlled at all. So the lines are edited where they sit, and an
    inline comment on an edited line is carried across.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    changed, inside = {}, False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(section + ":"):
            inside = True
            continue
        if inside and stripped and not line[:1].isspace():
            inside = False                     # a new top-level key ended the block
        if not inside:
            continue
        for field, value in values.items():
            if not stripped.startswith(field + ":"):
                continue
            indent = line[:len(line) - len(line.lstrip())]
            comment = ""
            if "#" in line:
                # Keep the text exactly, only normalising the gap before it.
                comment = "  # " + line.split("#", 1)[1].strip()
            lines[i] = "%s%s: %s%s\n" % (indent, field, _yaml_value(value), comment.rstrip())
            changed[field] = value
    if not changed:
        return {"error": "nothing named %s was found under '%s:' in the profile"
                         % (", ".join(values), section)}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
    return {"changed": changed}


def set_hardware(path: str, values: dict) -> dict:
    """Kept as its own name: the hardware panel is the commonest edit."""
    return set_section(path, "hardware", values)


PROFILE_TEMPLATE = """# {customer} -- created in the release studio.
# Version-controlled deliberately: a change to what this customer is licensed
# for, or how much of their machine we use, becomes a reviewable commit.
schema_version: 1

licence:
  customer: {customer}
  expires: {expires}
  frameworks: {frameworks}
  seats: {seats}

hardware:
  physical_cores: {cores}
  ram_gb: {ram}
  ctx_per_request: {ctx}

model: {model}

bundle: auto

runtime:
  max_audits_per_auditor: 2
  remediation_batch_size: 4
  remediation_timeout_sec: 1800
  ai_recommendations_default: true
  jwt_expiry_hours: 8
  locked: []                 # frameworks are never here; they are licence-controlled

build:
  compile_source: true
  encrypt_bundle: true
  run_tests: true
  run_sca: true
  fail_on_sca_severity: HIGH
  verify_models: true
  write_checksum: true
"""


def create_profile(directory: str, data: dict) -> dict:
    """Write a new customer profile, validating it before it is saved.

    Refuses to overwrite: a profile is what a customer is licensed for, and
    silently replacing one is how a site ends up entitled to something nobody
    decided to sell them.
    """
    name = re.sub(r"[^a-z0-9_-]+", "-", str(data.get("customer", "")).strip().lower()).strip("-")
    if not name:
        return {"error": "a customer name is required"}
    path = os.path.join(directory, name + ".yaml")
    if os.path.exists(path):
        return {"error": "%s.yaml already exists -- edit it rather than replacing it" % name}
    frameworks = [f for f in (data.get("frameworks") or []) if f]
    if not frameworks:
        return {"error": "a licence has to grant at least one framework"}
    body = PROFILE_TEMPLATE.format(
        customer=data.get("customer"), expires=data.get("expires"),
        frameworks=_yaml_value(frameworks), seats=int(data.get("seats") or 1),
        cores=int(data.get("physical_cores") or 8),
        ram=data.get("ram_gb") or 32,
        ctx=int(data.get("ctx_per_request") or 32768),
        model=data.get("model") or "google_gemma-4-E4B-it-Q4_K_M.gguf")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    try:
        load_profile(tmp)                 # never leave an invalid profile behind
    except Exception as exc:
        os.remove(tmp)
        return {"error": str(exc)[:400]}
    os.replace(tmp, path)
    return {"created": os.path.basename(path)}


def what_if(cores: int, ram_gb: float, ctx: int, per_auditor: int = 2) -> dict:
    """Size a machine without touching any profile -- the calculator."""
    from studio.sizing import _load
    s = _load().size_deployment(
        physical_cores=int(cores), total_ram_gb=float(ram_gb), model_gb=None,
        ctx_per_request=int(ctx), max_audits_per_auditor=int(per_auditor))
    return s.as_dict()


def known_versions(repo: str) -> list:
    """Tags in the product repository, newest first.

    The page offers these rather than leaving an operator to guess. Typing
    "v3.2" for "v3.24" produced a raw git message about ambiguous arguments and
    separating paths from revisions, which tells whoever reads it nothing they
    can act on.
    """
    import subprocess
    try:
        r = subprocess.run(["git", "-C", repo, "tag", "--sort=-creatordate"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return [t.strip() for t in (r.stdout or "").splitlines() if t.strip()][:25]


def _sizer(**kw):
    from studio.sizing import _load
    return _load().size_deployment(**kw).as_dict()


def _issue_for(profile: Profile) -> str:
    return issue(private_key_from_env(), customer=profile.licence.customer,
                 expires=profile.licence.expires,
                 frameworks=[f.value for f in profile.licence.frameworks],
                 seats=profile.licence.seats, tokens=profile.licence.tokens)


# ── the request handler ──────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    profiles_dir = os.path.join(ROOT, "profiles")
    repo = "."
    out_dir = "out"

    def log_message(self, fmt, *args):        # quiet; the page is the interface
        pass

    # -- plumbing --
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- routes --
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(PAGE, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError as exc:
                self._send(500, str(exc).encode(), "text/plain")
            return
        if path == "/api/profiles":
            self._json({"profiles": [profile_summary(p)
                                     for p in profile_paths(self.profiles_dir)],
                        "repo": os.path.abspath(self.repo),
                        "versions": known_versions(self.repo)})
            return
        if path.startswith("/api/build/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            self._json(job.as_dict() if job else {"error": "no such job"},
                       200 if job else 404)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()
        if path == "/api/plan":
            self._json(self._plan(body))
            return
        if path == "/api/build":
            self._json(self._build(body))
            return
        if path == "/api/size":
            try:
                self._json({"sizing": what_if(
                    body.get("cores", 8), body.get("ram_gb", 32),
                    body.get("ctx", 32768), body.get("per_auditor", 2))})
            except Exception as exc:
                self._json({"error": str(exc)[:200]})
            return
        if path == "/api/customer":
            res = create_profile(self.profiles_dir, body)
            if "created" in res:
                res["summary"] = profile_summary(
                    os.path.join(self.profiles_dir, res["created"]))
            self._json(res)
            return
        if path == "/api/licence":
            target = os.path.join(self.profiles_dir,
                                  os.path.basename(body.get("profile", "")))
            if not os.path.isfile(target):
                self._json({"error": "no such profile"}, 404)
                return
            # Frameworks are the commercial decision this whole tool exists to
            # carry, so an empty list is refused rather than written: a licence
            # granting nothing produces an installation that audits nothing.
            fields = {}
            if "frameworks" in body:
                fw = [f for f in (body.get("frameworks") or []) if f]
                if not fw:
                    self._json({"error": "a licence has to grant at least one framework"})
                    return
                fields["frameworks"] = fw
            for k in ("customer", "expires", "seats"):
                if k in body and str(body[k]).strip():
                    fields[k] = body[k]
            res = set_section(target, "licence", fields) if fields else {"changed": {}}
            gates = {k: bool(body[k]) for k in
                     ("compile_source", "encrypt_bundle", "run_tests", "run_sca",
                      "verify_models", "write_checksum") if k in body}
            if "fail_on_sca_severity" in body:
                gates["fail_on_sca_severity"] = str(body["fail_on_sca_severity"]).upper()
            if gates:
                g = set_section(target, "build", gates)
                res.setdefault("changed", {}).update(g.get("changed", {}))
            # Re-read rather than echo: what comes back is what the file says.
            summary = profile_summary(target)
            if not summary.get("valid"):
                res["error"] = "saved, but the profile no longer loads: %s" % summary.get("error")
            res["summary"] = summary
            self._json(res)
            return
        if path == "/api/hardware":
            target = os.path.join(self.profiles_dir,
                                  os.path.basename(body.get("profile", "")))
            if not os.path.isfile(target):
                self._json({"error": "no such profile"}, 404)
                return
            values = {k: body[k] for k in _HW_FIELDS if k in body}
            res = set_hardware(target, values)
            # Re-read it, so what comes back is what the file now says rather
            # than what was asked for.
            res["summary"] = profile_summary(target)
            self._json(res)
            return
        self._json({"error": "not found"}, 404)

    # -- actions --
    def _plan(self, body: dict) -> dict:
        """A dry run: decide the shape, size the machine, build nothing."""
        path = os.path.join(self.profiles_dir, os.path.basename(body.get("profile", "")))
        if not os.path.isfile(path):
            return {"error": "no such profile"}
        try:
            p = load_profile(path)
        except (ValidationError, yaml.YAMLError) as exc:
            return {"error": str(exc)[:400]}
        version = (body.get("version") or "").strip()
        previous = (body.get("previous") or "").strip() or None
        if not version:
            return {"error": "a version is required"}

        # An explicit choice from the page overrides the profile's default. Auto
        # still decides, and a patch it judges illegal is still refused: the
        # operator can ask for a patch, not overrule the reason one cannot work.
        chosen = (body.get("shape") or "").strip().lower()
        if chosen in ("patch", "full", "auto"):
            try:
                p = p.model_copy(update={"bundle": BundleShape(chosen)})
            except ValueError:
                pass
        shape, reason = p.bundle.value, ""
        base = p.patch_from or previous
        if p.bundle in (BundleShape.AUTO, BundleShape.PATCH) and base:
            try:
                d = patch_is_legal(self.repo, base, version)
            except PackagingError as exc:
                # git's own wording here is about ambiguous arguments and
                # separating paths from revisions. True, and useless to the
                # person who simply mistyped a version.
                tags = known_versions(self.repo)
                missing = [v for v in (base, version) if v not in tags]
                if missing and tags:
                    return {"error": "There is no version %s in the repository. "
                                     "Versions that exist: %s"
                                     % (" or ".join(missing), ", ".join(tags))}
                if not tags:
                    return {"error": "The product repository has no version tags yet, "
                                     "so there is nothing to compare against. Tag a "
                                     "release first (git tag -a v3.25) and the patch "
                                     "path becomes available."}
                return {"error": "Could not compare %s with %s: %s" % (base, version, exc)}
            if p.bundle == BundleShape.PATCH and not d.legal:
                return {"refused": True, "shape": "full", "reason": d.reason,
                        "blocking": d.blocking_changes}
            shape, reason = d.shape, d.reason
        elif not base:
            reason = "no previous version given"
        if p.bundle == BundleShape.PATCH and not base:
            return {"error": "a patch needs the version the customer is on"}
        from studio.executors import bundle_expectations
        return {"shape": shape, "reason": reason,
                "contents": bundle_expectations(shape, version),
                "summary": profile_summary(path)}

    def _build(self, body: dict) -> dict:
        path = os.path.join(self.profiles_dir, os.path.basename(body.get("profile", "")))
        if not os.path.isfile(path):
            return {"error": "no such profile"}
        version = (body.get("version") or "").strip()
        if not version:
            return {"error": "a version is required"}
        previous = (body.get("previous") or "").strip() or None
        try:
            profile = load_profile(path)
        except (ValidationError, yaml.YAMLError) as exc:
            return {"error": str(exc)[:400]}
        chosen = (body.get("shape") or "").strip().lower()
        if chosen in ("patch", "full", "auto"):
            try:
                profile = profile.model_copy(update={"bundle": BundleShape(chosen)})
            except ValueError:
                pass
        if profile.bundle == BundleShape.PATCH and not (profile.patch_from or previous):
            return {"error": "a patch needs the version the customer is on"}

        job = Job(uuid.uuid4().hex[:12])
        JOBS[job.id] = job
        repo, out_dir = self.repo, self.out_dir

        def worker():
            try:
                os.makedirs(out_dir, exist_ok=True)
                ctx = ChainContext(profile, version, os.path.abspath(out_dir))
                for step in run_chain(profile, ctx, repo=repo, version=version,
                                      previous=previous, issuer=_issue_for,
                                      sizer=_sizer):
                    job.add(step)
                    job.shape = ctx.shape
                    if not step.ok:
                        job.ok = False
                        break
                else:
                    job.ok = ctx.report.ok
                job.artifact = ctx.artifact
                job.licence_key = ctx.licence_key
                job.sha256 = ctx.sha256
            except LicenceError as exc:
                # The commonest first-run failure by some margin, and the least
                # obvious from a traceback.
                job.ok = False
                job.error = ("%s -- the signing key is read from the environment. "
                             "Set AUDITBOX_LICENCE_KEY before building." % exc)
            except Exception:
                job.ok = False
                job.error = traceback.format_exc(limit=4)
            finally:
                job.done = True

        threading.Thread(target=worker, daemon=True).start()
        return {"job": job.id}


def serve(port: int = 8770, profiles_dir: Optional[str] = None,
          repo: str = ".", out_dir: str = "out") -> None:
    Handler.profiles_dir = profiles_dir or os.path.join(ROOT, "profiles")
    Handler.repo = repo
    Handler.out_dir = out_dir
    # 127.0.0.1, never 0.0.0.0: this can start builds and hand back licence
    # keys, and it has no authentication.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("\nAuditBox Release Studio")
    print("   open   http://127.0.0.1:%d" % port)
    print("   repo   %s" % os.path.abspath(repo))
    print("   out    %s" % os.path.abspath(out_dir))
    print("\n   local only, no authentication. Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")
