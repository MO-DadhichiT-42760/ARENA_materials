"""Check CLI TLS initialization in fresh processes, without external downloads."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_python(source):
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_does_not_change_process_tls():
    run_python('''
import ssl
original = ssl.SSLContext
import scripts.prepare_chapter0
assert ssl.SSLContext is original
''')


def test_cli_configures_verified_system_tls_before_download_imports():
    # Stop at the first download import: no remote traffic or large data needed.
    # A fresh process matters because clients can retain SSLContext at import time.
    run_python('''
import builtins
import ssl
import sys
import truststore
from scripts.prepare_chapter0 import main

class DownloadsReached(Exception):
    pass

original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "datasets":
        assert ssl.SSLContext is truststore.SSLContext
        context = ssl.create_default_context()
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        import httpx
        import requests
        import urllib3.util.ssl_
        assert isinstance(httpx.create_ssl_context(), truststore.SSLContext)
        assert urllib3.util.ssl_.SSLContext is truststore.SSLContext
        raise DownloadsReached
    return original_import(name, *args, **kwargs)

builtins.__import__ = checked_import
sys.argv = ["prepare_chapter0.py"]
try:
    main()
except DownloadsReached:
    pass
else:
    raise AssertionError("No download clients imported")
''')
