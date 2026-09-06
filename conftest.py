"""Put the repository root on sys.path for the test suite.

Each test module used to do this for itself with a sys.path.insert computed
from __file__. That worked while they were run as scripts; under pytest the
import happens through the rootdir, so it belongs in one place.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
