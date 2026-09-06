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
