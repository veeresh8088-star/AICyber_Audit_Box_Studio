"""Command line for the release studio.

    studio validate profiles/stpi.yaml
    studio plan     profiles/stpi.yaml --version 3.24 --previous 3.23
    studio keygen   --out keys/
    studio licence  profiles/stpi.yaml

`plan` runs everything that does not touch Docker: it resolves the bundle
shape, sizes the customer's hardware and prints what a build would do. That is
deliberately the default way in -- a dry run costs seconds and catches an
illegal patch or an impossible model before anyone waits on an 8GB build.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import yaml
from pydantic import ValidationError

from studio.config import Profile, BundleShape
from studio.licensing import (
    generate_keypair, private_key_from_env, issue, LicenceError,
)
from studio.packaging import patch_is_legal, PackagingError, encrypt_bundle
from studio import executors as ex
from studio import pipeline as pl
from studio.sizing import size_for_profile


def load_profile(path: str) -> Profile:
    if not os.path.isfile(path):
        raise SystemExit(f"no profile at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return Profile(**raw)
    except ValidationError as exc:
        print(f"\n{path} is not a valid profile:\n", file=sys.stderr)
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err["loc"]) or "(root)"
            print(f"   {loc}: {err['msg']}", file=sys.stderr)
        raise SystemExit(2)


def cmd_validate(args) -> int:
    p = load_profile(args.profile)
    s = size_for_profile(p)
    print(f"\n{args.profile} is valid.\n")
    print(f"  customer    {p.licence.customer}")
    print(f"  frameworks  {', '.join(f.value for f in p.licence.frameworks)}")
    print(f"  expires     {p.licence.expires.isoformat()}  "
          f"({(p.licence.expires - date.today()).days} days)")
    print(f"  model       {p.model.value}")
    print(f"  hardware    {p.hardware.physical_cores} cores, {p.hardware.ram_gb}GB")
    print()
    print(f"  computed for that machine:")
    print(f"    -np {s.np_slots}   -c {s.shared_pool:,}   "
          f"({s.ctx_per_request:,} per request)")
    print(f"    {s.max_concurrent_audits} concurrent audits, "
          f"{s.max_audits_per_auditor} per auditor")
    print(f"    model {s.model_gb}GB + KV {s.np_slots * s.kv_gb_per_slot:.1f}GB "
          f"= {s.projected_llm_gb}GB of {s.total_ram_gb}GB "
          f"(headroom {s.headroom_gb}GB, limited by {s.limited_by})")
    if p.build.compile_source is False:
        print("\n  WARNING  compile_source is off: readable .py will ship to the customer.")
    return 0


def cmd_plan(args) -> int:
    p = load_profile(args.profile)
    print(f"\nplan for {p.licence.customer} - version {args.version}\n")

    shape = p.bundle.value
    base = p.patch_from or args.previous
    if p.bundle in (BundleShape.AUTO, BundleShape.PATCH) and base:
        try:
            d = patch_is_legal(args.repo, base, args.version)
        except PackagingError as exc:
            print(f"  cannot compare {base}..{args.version}: {exc}")
            return 3
        if p.bundle == BundleShape.PATCH and not d.legal:
            print(f"  REFUSED  a patch from {base} cannot apply")
            print(f"           {d.reason}")
            print(f"           build a full bundle instead.")
            return 3
        shape = d.shape
        print(f"  shape       {shape}  ({d.reason})")
    else:
        print(f"  shape       {shape}"
              + ("  (no previous version given)" if not base else ""))

    s = size_for_profile(p)
    print(f"  sizing      -np {s.np_slots}  -c {s.shared_pool:,}  "
          f"{s.max_concurrent_audits} concurrent audits")
    print(f"  gates       tests={'on' if p.build.run_tests else 'OFF'}  "
          f"sca={p.build.fail_on_sca_severity if p.build.run_sca else 'OFF'}  "
          f"compile={'on' if p.build.compile_source else 'OFF'}  "
          f"encrypt={'on' if p.build.encrypt_bundle else 'OFF'}")
    print("\n  nothing was built. This is a dry run.")
    return 0


def cmd_keygen(args) -> int:
    priv, pub = generate_keypair()
    os.makedirs(args.out, exist_ok=True)
    pub_path = os.path.join(args.out, "licence_public.pem")
    with open(pub_path, "wb") as fh:
        fh.write(pub)
    print("\nGenerated an Ed25519 licence keypair.\n")
    print(f"  public key written to {pub_path}")
    print("  ship this with the product; it can verify licences, never mint them.\n")
    print("  PRIVATE KEY -- store in your secret manager, never in git:\n")
    print(priv.decode())
    print("  export it for signing:  AUDITBOX_LICENCE_KEY=\"$(cat private.pem)\"")
    return 0


def cmd_licence(args) -> int:
    p = load_profile(args.profile)
    try:
        key = issue(
            private_key_from_env(),
            customer=p.licence.customer,
            expires=p.licence.expires,
            frameworks=[f.value for f in p.licence.frameworks],
            seats=p.licence.seats,
            tokens=p.licence.tokens,
        )
    except LicenceError as exc:
        print(f"\ncannot issue: {exc}", file=sys.stderr)
        return 2
    print(f"\n{p.licence.customer} - {', '.join(f.value for f in p.licence.frameworks)} "
          f"- expires {p.licence.expires.isoformat()}\n")
    print(key)
    return 0


def cmd_build(args) -> int:
    """Run the whole chain. Every gate must pass or nothing is produced.

    The sequence itself lives in studio.chain, shared with the web UI. Two front
    ends each assembling their own chain would drift, and the artifact somebody
    ships would be the one that skipped a gate.
    """
    from studio.chain import ChainContext, run_chain
    p = load_profile(args.profile)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ctx = ChainContext(p, args.version, out_dir)

    publisher = (ex.artifactory_publisher(args.artifactory, args.artifactory_repo)
                 if args.artifactory else None)
    for step in run_chain(p, ctx, repo=args.repo, version=args.version,
                          previous=args.previous, test_path=args.test_path,
                          scan_target=args.scan_target, issuer=_issue_for,
                          sizer=_sizer, publisher=publisher):
        mark = "skip" if step.skipped else ("ok" if step.ok else "FAIL")
        print(f"   {mark:>4}  {step.name:<16} {step.detail[:70]}")
        if not step.ok:
            print(ctx.report.summary())
            print(f"\nreport: {_write_report(ctx.report, out_dir, args.version)}")
            return 3

    print(ctx.report.summary())
    print(f"\nartifact: {ctx.artifact}")
    if ctx.licence_key:
        print(f"   licence:  {ctx.licence_key[:48]}...")
    print(f"   report:   {_write_report(ctx.report, out_dir, args.version)}")
    return 0


def cmd_ui(args) -> int:
    """Serve the local page for whoever is not going to type any of this."""
    from studio import web
    web.serve(port=args.port, repo=args.repo, out_dir=args.out)
    return 0


def _sizer(**kw):
    from studio.sizing import _load
    mod = _load()
    return mod.size_deployment(**kw).as_dict()


def _issue_for(profile) -> str:
    return issue(
        private_key_from_env(),
        customer=profile.licence.customer,
        expires=profile.licence.expires,
        frameworks=[f.value for f in profile.licence.frameworks],
        seats=profile.licence.seats,
        tokens=profile.licence.tokens,
    )


def _write_report(report, out_dir: str, version: str) -> str:
    path = os.path.join(out_dir, f"build-report-{version}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2, default=str)
    return path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="studio", description="AuditBox release studio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a profile and show its computed settings")
    v.add_argument("profile")
    v.set_defaults(func=cmd_validate)

    pl = sub.add_parser("plan", help="dry run: resolve shape and sizing, build nothing")
    pl.add_argument("profile")
    pl.add_argument("--version", required=True)
    pl.add_argument("--previous", default=None, help="version the customer is on")
    pl.add_argument("--repo", default=".", help="path to the product repository")
    pl.set_defaults(func=cmd_plan)

    k = sub.add_parser("keygen", help="generate a licence signing keypair")
    k.add_argument("--out", default="keys")
    k.set_defaults(func=cmd_keygen)

    b = sub.add_parser("build", help="run the whole chain: gates, bundle, licence, publish")
    b.add_argument("profile")
    b.add_argument("--version", required=True)
    b.add_argument("--previous", default=None)
    b.add_argument("--repo", default=".", help="path to the product repository")
    b.add_argument("--out", default="out", help="where artifacts are written")
    b.add_argument("--scan-target", default=None, help="image or path for the SCA scan")
    b.add_argument("--test-path", nargs="*", default=None,
                   help="test paths to run. The product's tests/ mixes pytest files "
                        "with runnable scripts that sys.exit at import, which pytest "
                        "cannot collect -- name the collectable ones explicitly.")
    b.add_argument("--artifactory", default=os.environ.get("ARTIFACTORY_URL"),
                   help="base URL; omitted, the artifact is left locally")
    b.add_argument("--artifactory-repo", default="auditbox/releases")
    b.set_defaults(func=cmd_build)

    u = sub.add_parser("ui", help="open the local page instead of typing commands")
    u.add_argument("--port", type=int, default=8770)
    u.add_argument("--repo", default=".", help="path to the product repository")
    u.add_argument("--out", default="out", help="where artifacts are written")
    u.set_defaults(func=cmd_ui)

    li = sub.add_parser("licence", help="issue a signed licence key for a profile")
    li.add_argument("profile")
    li.set_defaults(func=cmd_licence)
    return ap


def main(argv=None) -> int:
    # The operator running this is on a Windows console, which is cp1252 more
    # often than not. Anything outside that map either raised UnicodeEncodeError
    # or printed as a literal "→" in the middle of the invalid-profile
    # report -- exactly the message somebody non-technical has to read. The
    # printed text is ASCII now; this is the belt for a customer name, path or
    # profile field that is not.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass                              # not reconfigurable; ASCII output still prints
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
