"""Bridge to the product's deployment_sizing module.

The arithmetic deliberately lives in the PRODUCT, not here: the launcher needs
it at boot and this tool needs it at build time, and two copies drift. That
already happened once -- run_all.bat assumed a 4.5GB model while shipping an
11.8GB one, and handed out slots against memory that did not exist.

So this module only locates the product and calls it. If the path is wrong it
says so plainly rather than quietly falling back to a second implementation,
because a silent fallback is how the two versions diverge again.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

from studio.config import Profile

_ENV_VAR = "AUDITBOX_PRODUCT_PATH"
_DEFAULT_GUESSES = (
    os.path.join(os.path.expanduser("~"), "Desktop", "Local_audit_box_full", "audit test_box"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "audit test_box"),
)


class ProductNotFound(Exception):
    pass


def _find_product() -> str:
    explicit = os.environ.get(_ENV_VAR, "").strip()
    candidates = [explicit] if explicit else list(_DEFAULT_GUESSES)
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "src", "core", "deployment_sizing.py")):
            return os.path.abspath(path)
    raise ProductNotFound(
        f"could not find the product repository. Set {_ENV_VAR} to the checkout "
        f"containing src/core/deployment_sizing.py -- the sizing arithmetic is "
        f"owned there so the launcher and this tool cannot disagree."
    )


def _load():
    root = _find_product()
    spec = importlib.util.spec_from_file_location(
        "auditbox_deployment_sizing",
        os.path.join(root, "src", "core", "deployment_sizing.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before executing. @dataclass resolves its own module through
    # sys.modules[cls.__module__], so a module loaded by path alone raises
    # AttributeError: 'NoneType' object has no attribute '__dict__' the moment
    # it defines one -- which deployment_sizing does.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def size_for_profile(profile: Profile, model_gb: Optional[float] = None):
    """Size a customer's machine from their profile.

    `available_ram_gb` is deliberately not passed: at build time "free RAM"
    means the build machine's, which has nothing to do with the customer's box.
    Only the total they told us is planned against.
    """
    mod = _load()
    resolved = model_gb
    if resolved is None:
        try:
            root = _find_product()
            candidate = os.path.join(root, profile.model.value)
            resolved = mod.model_size_gb(candidate) if os.path.exists(candidate) else None
        except ProductNotFound:
            resolved = None
    return mod.size_deployment(
        physical_cores=profile.hardware.physical_cores,
        total_ram_gb=profile.hardware.ram_gb,
        model_gb=resolved,
        ctx_per_request=profile.hardware.ctx_per_request,
        max_audits_per_auditor=profile.runtime.max_audits_per_auditor,
    )
