# AuditBox Release Studio

Builds, licenses and packages AICyberAuditBox for a customer.

Separate from the product repository on purpose: this tool never ships to a
customer, and keeping it out of the product makes that structurally true rather
than a rule someone has to remember.

## What it does

    studio validate profiles/stpi.yaml            check a profile, show its computed settings
    studio plan     profiles/stpi.yaml --version 3.24 --previous 3.23
    studio keygen   --out keys/                   generate a licence signing keypair
    studio licence  profiles/stpi.yaml            issue a signed licence key

`plan` is a dry run: it resolves the bundle shape and sizes the customer's
hardware without touching Docker. Run it first — it takes seconds and catches an
illegal patch or an impossible model before anyone waits on an 8GB build.

## Customer profiles

One YAML file per customer in `profiles/`, version-controlled, so a change to
what somebody is licensed for is a reviewable commit with an author and a date.

Settings fall into three tiers, and the split is commercial, not technical:

| tier | who may change it | examples |
|---|---|---|
| build-time | nobody, once shipped | licensed frameworks, which model ships, compiled or not |
| install-time | the deployment engineer | hardware sizing, ports, secrets |
| runtime | the customer's admin | concurrency, timeouts, AI recommendations |

Frameworks are never a runtime setting. They live in the signed licence and are
enforced server-side, because a customer who can flip PQC on from a settings
page is a customer you cannot sell PQC to.

## Sizing

The arithmetic lives in the **product**, at `src/core/deployment_sizing.py`, and
this tool calls it. Two copies would drift, and already did once: the launcher
assumed a 4.5GB model while shipping an 11.8GB one, handing out KV slots against
memory that did not exist. Point `AUDITBOX_PRODUCT_PATH` at the checkout.

## Keys

    studio keygen --out keys/

Writes the public key (ship it with the product; it verifies licences and cannot
mint them) and prints the private key to store in your secret manager. Export it
as `AUDITBOX_LICENCE_KEY` to sign. It is never read from a file in this repo.

## Tests

    python tests/test_config.py
    python tests/test_licensing.py
    python tests/test_packaging.py
    python tests/test_pipeline.py

99 checks. Each module is tested in isolation with injected fakes, so the chain
is verifiable without Docker, a model, or a network.

## Not built yet

- Docker build execution and the Artifactory publisher (both need credentials
  and an instance to target)
- Nuitka compilation — `compile_source` is honoured as a gate, but the compiler
  step is not implemented; leaving it on with no compiler will fail the build
  rather than silently shipping readable source
- The web console, for issuing licences without the command line
