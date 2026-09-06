"""The page has to work for whoever was never going to type the commands.

Serving it from the standard library is deliberate: the studio has three
dependencies and each has to install on a build machine that may be offline, so
a web framework for a single-user local page would be a fourth for no gain. For
the same reason the page carries its own CSS and JS and loads nothing remote --
a CDN reference works on the machine it was written on and fails at the customer
site, which is the whole failure mode this product exists to avoid.
"""
import json
import os
import re
import threading

import pytest

from studio import web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_page_exists_and_is_self_contained():
    with open(web.PAGE, encoding="utf-8") as fh:
        html = fh.read()
    remote = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    assert not remote, "the page loads something remote: %s" % remote
    assert "<title>" in html and "/api/profiles" in html


def test_profiles_are_summarised_for_the_page():
    paths = web.profile_paths(os.path.join(ROOT, "profiles"))
    assert paths, "no profiles on disk"
    s = web.profile_summary(paths[0])
    assert s["valid"], s
    for key in ("customer", "frameworks", "expires", "model", "sizing", "gates"):
        assert key in s, key
    # The numbers an operator decides on, not raw profile fields.
    assert s["sizing"]["np_slots"] >= 1
    assert s["sizing"]["max_concurrent_audits"] >= 1


def test_a_broken_profile_is_reported_not_raised(tmp_path):
    """One unparseable file must not take the whole list down."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("licence:\n  customer: ''\n  frameworks: []\n")
    s = web.profile_summary(str(bad))
    assert s["valid"] is False
    assert s["error"]


def test_profile_paths_of_a_missing_directory_is_empty():
    assert web.profile_paths("/definitely/not/here") == []


# -- the job object the page polls -------------------------------------------

class _Step:
    def __init__(self, name, ok, detail, seconds=0.1, skipped=False):
        self.name, self.ok, self.detail = name, ok, detail
        self.seconds, self.skipped = seconds, skipped


def test_a_job_reports_steps_as_they_land():
    job = web.Job("abc")
    assert job.as_dict()["steps"] == []
    job.add(_Step("tests", True, "258 passed"))
    job.add(_Step("sca", False, "1 HIGH: pillow"))
    d = job.as_dict()
    assert [s["name"] for s in d["steps"]] == ["tests", "sca"]
    assert d["steps"][1]["ok"] is False
    assert d["done"] is False


def test_a_job_is_safe_to_read_while_it_is_written():
    """The page polls every second while the chain writes from its own thread."""
    job = web.Job("race")
    stop = threading.Event()

    def writer():
        for i in range(400):
            job.add(_Step("step%d" % i, True, "detail"))
        stop.set()

    t = threading.Thread(target=writer)
    t.start()
    while not stop.is_set():
        json.dumps(job.as_dict())          # would raise if the list mutated mid-read
    t.join()
    assert len(job.as_dict()["steps"]) == 400


def test_the_server_binds_to_loopback_only():
    """It can start builds and hand back licence keys, and has no auth."""
    import inspect
    # The bind call itself, not the whole source: the comment above it mentions
    # 0.0.0.0 precisely to say it is not used, and a naive search matched that.
    bind = [l for l in inspect.getsource(web.serve).splitlines()
            if "ThreadingHTTPServer(" in l and not l.strip().startswith("#")]
    assert len(bind) == 1, bind
    assert '"127.0.0.1"' in bind[0], bind[0]
    assert "0.0.0.0" not in bind[0], bind[0]


# -- the machine the customer actually has -----------------------------------

def test_sizing_follows_the_hardware_it_is_given():
    """Different customers, different boxes. The numbers have to move with them."""
    small = web.what_if(4, 16, 32768)
    big = web.what_if(64, 256, 32768)
    assert small["np_slots"] < big["np_slots"]
    assert small["max_concurrent_audits"] <= big["max_concurrent_audits"]
    # A small box is held back by memory, a wide one by cores.
    assert small["limited_by"] == "ram", small
    assert big["limited_by"] == "cores", big


def test_hardware_is_edited_without_losing_the_rest_of_the_file(tmp_path):
    """A yaml round trip would drop the comments the profiles exist to carry."""
    p = tmp_path / "cust.yaml"
    p.write_text(
        "# Why this customer is licensed for what they are.\n"
        "licence:\n  customer: ACME\n"
        "hardware:\n"
        "  physical_cores: 8       # their box\n"
        "  ram_gb: 32\n"
        "  ctx_per_request: 32768\n"
        "model: gemma-4-12B-it-Q8_0.gguf\n")
    res = web.set_hardware(str(p), {"physical_cores": 32, "ram_gb": 125})
    body = p.read_text()
    assert res["changed"] == {"physical_cores": 32, "ram_gb": 125}
    assert "physical_cores: 32" in body
    assert "ram_gb: 125" in body
    assert "# Why this customer is licensed" in body, "the comment was lost"
    assert "# their box" in body, "the inline comment was lost"
    assert "model: gemma-4-12B-it-Q8_0.gguf" in body, "an unrelated key was lost"
    assert "ctx_per_request: 32768" in body, "an untouched field changed"


def test_editing_a_profile_with_no_hardware_block_says_so(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("licence:\n  customer: ACME\n")
    assert "error" in web.set_hardware(str(p), {"physical_cores": 8})


# -- versions the operator can act on ----------------------------------------
# Typing v3.2 for v3.24 produced git's own words: "fatal: ambiguous argument
# 'v3.2..v3.25': unknown revision or path not in the working tree. Use '--' to
# separate paths from revisions". Accurate, and no use at all to the person who
# simply mistyped a version -- which is the person this page is for.

def test_known_versions_reads_the_repository_tags(tmp_path):
    import subprocess
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)

    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    git("add", "-A"); git("commit", "-qm", "1"); git("tag", "v3.24")
    assert web.known_versions(str(repo)) == ["v3.24"]


def test_a_repository_with_no_tags_returns_nothing(tmp_path):
    """Not an error: the page says the patch path is unavailable until tagged."""
    import subprocess
    repo = tmp_path / "empty"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], capture_output=True)
    assert web.known_versions(str(repo)) == []


def test_known_versions_of_a_non_repository_is_empty_not_a_crash(tmp_path):
    assert web.known_versions(str(tmp_path)) == []


# -- customers, created and amended from the page ----------------------------
# What a site is licensed for is the commercial decision this whole tool exists
# to carry. Editing it by hand in YAML is how a customer ends up entitled to
# something nobody decided to sell them.

def _profiles_dir(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    return str(d)


def test_a_customer_can_be_created_and_reads_back(tmp_path):
    d = _profiles_dir(tmp_path)
    res = web.create_profile(d, {"customer": "Acme Bank", "expires": "2027-01-01",
                                 "frameworks": ["ISO27001", "VAPT"], "seats": 5,
                                 "physical_cores": 16, "ram_gb": 64})
    assert res["created"] == "acme-bank.yaml", res
    s = web.profile_summary(str(tmp_path / "profiles" / "acme-bank.yaml"))
    assert s["valid"] and s["customer"] == "Acme Bank"
    assert s["frameworks"] == ["ISO27001", "VAPT"]
    assert s["seats"] == 5


def test_a_licence_granting_nothing_is_refused(tmp_path):
    """An installation licensed for no framework audits nothing."""
    res = web.create_profile(_profiles_dir(tmp_path),
                             {"customer": "X", "expires": "2027-01-01", "frameworks": []})
    assert "at least one framework" in res["error"]


def test_creating_never_overwrites_an_existing_customer(tmp_path):
    d = _profiles_dir(tmp_path)
    data = {"customer": "Acme", "expires": "2027-01-01", "frameworks": ["PQC"]}
    assert "created" in web.create_profile(d, data)
    again = web.create_profile(d, data)
    assert "already exists" in again["error"]


def test_an_invalid_profile_is_never_left_on_disk(tmp_path):
    """The file is validated before it replaces anything."""
    d = _profiles_dir(tmp_path)
    res = web.create_profile(d, {"customer": "Bad", "expires": "not-a-date",
                                 "frameworks": ["PQC"]})
    assert "error" in res
    assert os.listdir(d) == [], os.listdir(d)


def test_entitlements_can_be_changed_after_the_fact(tmp_path):
    d = _profiles_dir(tmp_path)
    web.create_profile(d, {"customer": "Acme", "expires": "2027-01-01",
                           "frameworks": ["ISO27001", "VAPT"]})
    path = os.path.join(d, "acme.yaml")
    res = web.set_section(path, "licence", {"frameworks": ["PQC"], "seats": 25})
    assert res["changed"]["frameworks"] == ["PQC"]
    s = web.profile_summary(path)
    assert s["frameworks"] == ["PQC"] and s["seats"] == 25


def test_build_gates_can_be_turned_off_deliberately(tmp_path):
    d = _profiles_dir(tmp_path)
    web.create_profile(d, {"customer": "Acme", "expires": "2027-01-01", "frameworks": ["PQC"]})
    path = os.path.join(d, "acme.yaml")
    web.set_section(path, "build", {"compile_source": False, "run_sca": False})
    g = web.profile_summary(path)["gates"]
    assert g["compile"] is False and g["sca"] is False
    assert g["tests"] is True, "an untouched gate changed"


def test_booleans_and_lists_are_written_as_yaml_not_python(tmp_path):
    """True/['PQC'] would not parse back; true/[PQC] does."""
    assert web._yaml_value(True) == "true"
    assert web._yaml_value(False) == "false"
    assert web._yaml_value(["PQC", "VAPT"]) == "[PQC, VAPT]"


# -- a page newer than the server running it ---------------------------------
# The page is read from disk on every request; the routes are whatever the
# process loaded at start. So editing the studio and reloading the browser
# showed new buttons wired to endpoints that answered "not found", which reads
# as a broken feature rather than a server that needs restarting.

def test_the_page_and_the_server_agree_on_the_api_version():
    """If a route was added, both numbers move. This test is the reminder."""
    import re as _re
    with open(web.PAGE, encoding="utf-8") as fh:
        html = fh.read()
    m = _re.search(r"const PAGE_API = (\d+)", html)
    assert m, "the page no longer declares PAGE_API"
    assert int(m.group(1)) == web.API_VERSION, (
        "studio/web.py API_VERSION=%d but the page expects %s -- bump both when "
        "adding a route, or the stale-server banner lies" % (web.API_VERSION, m.group(1)))


def test_the_profiles_response_carries_what_the_banner_needs():
    """api_version and is_git_repo are what the page uses to explain itself."""
    import inspect
    src = inspect.getsource(web.Handler.do_GET)
    assert "api_version" in src and "is_git_repo" in src


def test_every_bundle_shape_is_explained_on_the_page():
    """"Patch only" tells an operator nothing about what is patched."""
    with open(web.PAGE, encoding="utf-8") as fh:
        html = fh.read()
    for word in ("application layer", "model weights", "2 GB", "8 GB"):
        assert word in html, "the shape explanation lost: %s" % word
    # The buttons say what happens, not what the code calls it.
    assert "App code only" in html and "Everything" in html


# -- the frameworks on offer are the ones that exist -------------------------
# The page carried a hand-written list. It offered PCIDSS and HIPAA, which this
# product has never audited, and left out DPDP, BCMS and XBOM, which it does.
# Ticking an invented one produced a profile that failed validation; leaving out
# a real one meant a customer could not be sold something we can deliver.

def test_the_page_does_not_carry_its_own_framework_list():
    with open(web.PAGE, encoding="utf-8") as fh:
        html = fh.read()
    # Comments are excluded: the one above the fetch names PCIDSS and HIPAA
    # precisely to record that they do not exist, and matching that is how the
    # first version of this test failed.
    code = "\n".join(l for l in html.splitlines() if not l.strip().startswith("//"))
    for invented in ("PCIDSS", "HIPAA"):
        assert invented not in code, "%s is on the page and does not exist" % invented
    assert "let FRAMEWORKS = []" in code, "the list is hardcoded again"


def test_every_framework_is_labelled_for_a_customer():
    """An operator is deciding what a site is sold, not reading an enum."""
    from studio.config import Framework, FRAMEWORK_LABELS
    for f in Framework:
        assert f.value in FRAMEWORK_LABELS, "%s has no customer-facing name" % f.value
        name, about, controls = FRAMEWORK_LABELS[f.value]
        assert name and about and controls > 0, (f.value, name, about, controls)


def test_the_labelled_control_counts_match_the_product():
    """217 across eight frameworks -- the number the product reports itself."""
    from studio.config import FRAMEWORK_LABELS
    assert sum(c for _, _, c in FRAMEWORK_LABELS.values()) == 217


def test_a_framework_the_product_has_can_be_licensed():
    """NIST and XBOM were absent from the schema, so neither could be sold."""
    from studio.config import Framework
    have = {f.value for f in Framework}
    for expected in ("ISO27001", "VAPT", "PQC", "SOC2", "DPDP", "BCMS", "NIST", "XBOM"):
        assert expected in have, "%s cannot be licensed" % expected


def test_an_invented_framework_is_still_refused(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    res = web.create_profile(str(d), {"customer": "X", "expires": "2027-01-01",
                                      "frameworks": ["HIPAA"]})
    assert "error" in res, "HIPAA was accepted"
