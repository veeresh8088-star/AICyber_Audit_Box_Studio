"""Customer release profiles: what a build is configured with.

One YAML file per customer, version-controlled, so a change to somebody's
licensed features or concurrency is a reviewable commit with an author and a
date rather than a setting someone remembers changing.

The schema draws a line that matters commercially:

  build-time   licensed frameworks, which model ships, compiled or not.
               Fixed when the bundle is cut. A customer cannot change these,
               which is the whole basis for selling PQC separately from ISO.
  install-time hardware sizing, ports, secrets. Set once by the deployment
               engineer, because the real machine may differ from the profile.
  runtime      concurrency, timeouts, AI recommendations. The customer's admin
               may change these freely; they only affect their own box.

Anything a customer may flip at runtime must be harmless to flip. Frameworks
are not, so they live in the licence and are enforced server-side.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Framework(str, Enum):
    """Sellable audit frameworks. These are the entitlement unit.

    All eight the product implements, checked against its own control set
    rather than assumed: 217 controls across ISO 27001 (93), SOC 2 (33), X-BOM
    (23), NIST CSF (22), DPDP/GDPR (15), VAPT (15), PQC (12) and BCMS (4).
    NIST and XBOM were missing here, so a customer could not be sold two things
    the product can actually audit.
    """
    ISO27001 = "ISO27001"
    VAPT = "VAPT"
    PQC = "PQC"
    SOC2 = "SOC2"
    DPDP = "DPDP"
    BCMS = "BCMS"
    NIST = "NIST"
    XBOM = "XBOM"


# What each one is called in front of a customer, and how much of the control
# set it grants. An operator ticking boxes is deciding what a site is sold, so
# the page shows this rather than a six-letter enum name.
FRAMEWORK_LABELS = {
    "ISO27001": ("ISO/IEC 27001:2022", "Information Security Management", 93),
    "NIST":     ("NIST CSF 2.0", "Cybersecurity Framework Core", 22),
    "DPDP":     ("DPDP / GDPR", "Digital Data Protection and EU GDPR", 15),
    "SOC2":     ("SOC 2 Type II", "System and Organization Controls", 33),
    "BCMS":     ("ISO 22301 BCMS", "Business Continuity Management", 4),
    "XBOM":     ("X-BOM / SBOM", "Software Bill of Materials", 23),
    "VAPT":     ("VAPT", "Vulnerability Assessment and Penetration Testing", 15),
    "PQC":      ("PQC", "Post-Quantum Cryptography Readiness", 12),
}


class ModelChoice(str, Enum):
    """Which .gguf ships in the bundle.

    The value is the filename, because the launcher and the sizing calculator
    both need the real file: sizing measures it on disk, and guessing its size
    is what caused a 4-core box to page for 29 minutes on a single control.
    """
    GEMMA4_12B_Q8 = "gemma-4-12B-it-Q8_0.gguf"
    GEMMA4_E4B_Q4 = "google_gemma-4-E4B-it-Q4_K_M.gguf"
    GEMMA2_9B_Q8 = "gemma-2-9b-it-Q8_0.gguf"


class BundleShape(str, Enum):
    AUTO = "auto"      # decide from what changed since the previous version
    FULL = "full"      # every image, ~8GB, first install on an air-gapped site
    PATCH = "patch"    # src/ and config/ only, ~6MB, rebuilt on their image


class Hardware(BaseModel):
    """The customer's machine, as told to us.

    Sizing is computed from this at build time so the bundle arrives correctly
    configured. The launcher still re-measures on the real machine, because a
    customer may deploy somewhere other than they said.
    """
    physical_cores: int = Field(..., ge=1, le=512)
    ram_gb: float = Field(..., gt=0, le=4096)
    ctx_per_request: int = Field(32768, ge=4096, le=131072)

    @field_validator("ctx_per_request")
    @classmethod
    def _power_of_two_ish(cls, v: int) -> int:
        if v % 1024:
            raise ValueError("ctx_per_request must be a multiple of 1024")
        return v


class RuntimeSettings(BaseModel):
    """Defaults the bundle ships with, and whether the customer may change them.

    `locked` names settings the admin UI must not expose. Frameworks are never
    here at all -- they are licence-controlled, not a setting.
    """
    max_concurrent_audits: Optional[int] = Field(None, ge=1, le=64)
    max_audits_per_auditor: int = Field(2, ge=1, le=16)
    remediation_batch_size: int = Field(4, ge=1, le=32)
    remediation_timeout_sec: int = Field(1800, ge=60, le=7200)
    ai_recommendations_default: bool = True
    jwt_expiry_hours: int = Field(8, ge=1, le=720)
    locked: List[str] = Field(default_factory=list)

    @field_validator("locked")
    @classmethod
    def _known_keys(cls, v: List[str]) -> List[str]:
        allowed = {
            "max_concurrent_audits", "max_audits_per_auditor",
            "remediation_batch_size", "remediation_timeout_sec",
            "ai_recommendations_default", "jwt_expiry_hours",
        }
        unknown = sorted(set(v) - allowed)
        if unknown:
            raise ValueError(f"cannot lock unknown setting(s): {', '.join(unknown)}")
        return v


class Licence(BaseModel):
    """What the customer bought.

    `frameworks` is the entitlement. It is signed into the licence key and
    enforced server-side, so a PQC-only customer cannot reach ISO by calling the
    API directly -- which a UI-only check would not prevent.
    """
    customer: str = Field(..., min_length=2, max_length=120)
    expires: date
    frameworks: List[Framework] = Field(..., min_length=1)
    seats: int = Field(5, ge=1, le=1000)
    tokens: Optional[int] = Field(None, ge=0)

    @field_validator("frameworks")
    @classmethod
    def _no_duplicates(cls, v: List[Framework]) -> List[Framework]:
        if len(set(v)) != len(v):
            raise ValueError("frameworks contains duplicates")
        return v


class BuildOptions(BaseModel):
    compile_source: bool = True     # native modules, no .py in the image
    encrypt_bundle: bool = True
    run_tests: bool = True
    run_sca: bool = True
    fail_on_sca_severity: str = "HIGH"
    # Confirms the model weights are actually inside the LLM image before it
    # ships. verify_images_tar proves the image TAG made it into the tar; it
    # cannot see inside the layers, so a cached layer or a COPY that silently
    # did nothing still produces an image that loads, starts, and then fails on
    # the customer's first inference.
    verify_models: bool = True
    # sha256 of the finished artifact, written beside it. A truncated multi-GB
    # transfer usually still opens as a tar.
    write_checksum: bool = True

    @field_validator("fail_on_sca_severity")
    @classmethod
    def _severity(cls, v: str) -> str:
        allowed = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        u = v.upper()
        if u not in allowed:
            raise ValueError(f"fail_on_sca_severity must be one of {sorted(allowed)}")
        return u


class Profile(BaseModel):
    """One customer's complete release configuration."""
    schema_version: int = Field(1, ge=1, le=1)
    licence: Licence
    hardware: Hardware
    model: ModelChoice
    bundle: BundleShape = BundleShape.AUTO
    patch_from: Optional[str] = None
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    build: BuildOptions = Field(default_factory=BuildOptions)

    @model_validator(mode="after")
    def _patch_needs_a_base(self) -> "Profile":
        if self.bundle == BundleShape.PATCH and not self.patch_from:
            raise ValueError("bundle: patch requires patch_from (the version to patch from)")
        if self.bundle != BundleShape.PATCH and self.patch_from:
            raise ValueError("patch_from is only meaningful when bundle is 'patch'")
        return self

    @model_validator(mode="after")
    def _12b_needs_the_ram(self) -> "Profile":
        """Refuse a configuration that cannot run, at build time.

        A 12B Q8 model is 11.8GB. Shipping it to a 16GB machine leaves nothing
        for the KV cache, the OS and Postgres -- the server pages and every
        audit crawls. Better to fail the build than to ship it and have the
        customer discover it.
        """
        if self.model == ModelChoice.GEMMA4_12B_Q8 and self.hardware.ram_gb < 24:
            raise ValueError(
                f"model {self.model.value} needs about 12GB for weights alone; "
                f"{self.hardware.ram_gb}GB leaves no room for KV cache and the OS. "
                f"Use {ModelChoice.GEMMA4_E4B_Q4.value} on this hardware."
            )
        return self
