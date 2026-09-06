"""A profile must refuse configurations that cannot work, at build time.

The point of validating here is that the failure lands on the person cutting
the release, who can fix it, rather than on the customer three weeks later.
"""
import os
from datetime import date, timedelta

import pytest
import yaml
from pydantic import ValidationError

from studio.config import Profile, Framework, ModelChoice, BundleShape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUTURE = (date.today() + timedelta(days=200)).isoformat()


def prof(**over):
    """A valid profile, with whatever the test wants to change about it."""
    base = dict(
        licence=dict(customer="ACME", expires=FUTURE, frameworks=["PQC"]),
        hardware=dict(physical_cores=32, ram_gb=125),
        model="google_gemma-4-E4B-it-Q4_K_M.gguf",
    )
    base.update(over)
    return Profile(**base)


def _shipped(name):
    with open(os.path.join(ROOT, "profiles", name), encoding="utf-8") as fh:
        return Profile(**yaml.safe_load(fh))


# -- the real profiles on disk parse -----------------------------------------

@pytest.mark.parametrize("name", ["stpi.yaml", "smallsite.yaml"])
def test_shipped_profile_is_valid(name):
    assert _shipped(name).licence.customer != ""


def test_stpi_is_pqc_only():
    """The customer this was written for buys PQC and must get only PQC."""
    assert _shipped("stpi.yaml").licence.frameworks == [Framework.PQC]


# -- a model that cannot fit is refused before it ships ----------------------

def test_a_model_too_large_for_the_box_is_refused():
    with pytest.raises(ValidationError):
        prof(model="gemma-4-12B-it-Q8_0.gguf",
             hardware=dict(physical_cores=8, ram_gb=16))


def test_a_model_that_fits_is_allowed():
    assert prof(model="gemma-4-12B-it-Q8_0.gguf",
                hardware=dict(physical_cores=8, ram_gb=32)
                ).model == ModelChoice.GEMMA4_12B_Q8
    assert prof(model="google_gemma-4-E4B-it-Q4_K_M.gguf",
                hardware=dict(physical_cores=4, ram_gb=16)).hardware.ram_gb == 16


# -- a patch needs a base, and a base needs a patch --------------------------

def test_patch_without_a_base_is_refused():
    with pytest.raises(ValidationError):
        prof(bundle="patch")


def test_a_base_without_a_patch_is_refused():
    """patch_from on a full bundle means somebody misunderstood the setting."""
    with pytest.raises(ValidationError):
        prof(bundle="full", patch_from="3.22")


def test_patch_with_a_base_is_valid():
    assert prof(bundle="patch", patch_from="3.22").patch_from == "3.22"


# -- a licence must grant something ------------------------------------------

@pytest.mark.parametrize("licence", [
    pytest.param(dict(customer="X", expires=FUTURE, frameworks=[]), id="no-frameworks"),
    pytest.param(dict(customer="X", expires=FUTURE, frameworks=["PQC", "PQC"]), id="duplicate"),
    pytest.param(dict(customer="X", expires=FUTURE, frameworks=["NOTREAL"]), id="unknown"),
    pytest.param(dict(customer="", expires=FUTURE, frameworks=["PQC"]), id="empty-customer"),
])
def test_meaningless_licence_is_refused(licence):
    with pytest.raises(ValidationError):
        prof(licence=licence)


# -- hardware must be plausible ----------------------------------------------

@pytest.mark.parametrize("hardware", [
    pytest.param(dict(physical_cores=0, ram_gb=64), id="zero-cores"),
    pytest.param(dict(physical_cores=8, ram_gb=0), id="zero-RAM"),
    pytest.param(dict(physical_cores=8, ram_gb=64, ctx_per_request=30000),
                 id="context-not-a-multiple-of-1024"),
])
def test_implausible_hardware_is_refused(hardware):
    with pytest.raises(ValidationError):
        prof(hardware=hardware)


def test_a_legal_context_size_is_accepted():
    assert prof(hardware=dict(physical_cores=8, ram_gb=64, ctx_per_request=65536)
                ).hardware.ctx_per_request == 65536


# -- frameworks can never be a customer-adjustable setting -------------------

def test_frameworks_cannot_be_declared_customer_adjustable():
    """What was licensed is decided at build time, by us, and never moves."""
    with pytest.raises(ValidationError):
        prof(runtime=dict(locked=["frameworks"]))


def test_locking_a_real_setting_is_fine():
    assert prof(runtime=dict(locked=["max_concurrent_audits"])
                ).runtime.locked == ["max_concurrent_audits"]


# -- build gates are validated -----------------------------------------------

def test_a_nonsense_severity_is_refused():
    with pytest.raises(ValidationError):
        prof(build=dict(fail_on_sca_severity="SPICY"))


def test_severity_is_normalised_to_upper_case():
    assert prof(build=dict(fail_on_sca_severity="critical")
                ).build.fail_on_sca_severity == "CRITICAL"


def test_compilation_can_be_turned_off_explicitly():
    """Allowed, but it has to be a deliberate line in the profile."""
    assert prof(build=dict(compile_source=False)).build.compile_source is False


# -- defaults are the safe ones ----------------------------------------------

def test_defaults_are_the_safe_ones():
    """Every gate is on unless somebody writes down that it should not be."""
    d = prof()
    assert d.build.compile_source is True
    assert d.build.encrypt_bundle is True
    assert d.build.run_tests is True
    assert d.build.run_sca is True
    assert d.build.fail_on_sca_severity == "HIGH"
    assert d.bundle == BundleShape.AUTO
