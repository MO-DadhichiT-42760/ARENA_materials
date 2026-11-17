"""Check CLI TLS initialization in fresh processes, without external downloads."""

import os
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


HUB_MODE_CHECK = """
import sys
import scripts.prepare_chapter0 as prepare

def record(*args, **kwargs):
    # Library config is read at import time, so check what the clients actually loaded.
    import datasets.config
    import huggingface_hub.constants
    print("HUB_OFFLINE", datasets.config.HF_HUB_OFFLINE, huggingface_hub.constants.HF_HUB_OFFLINE)

prepare.prepare_core = record
prepare.prepare_weights = lambda *a, **k: None
prepare.prepare_celeba = lambda *a, **k: None
prepare.main(sys.argv[1:])
"""


def hub_mode(*args, **env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    result = subprocess.run(
        [sys.executable, "-c", HUB_MODE_CHECK, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**clean, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_core_preparation_keeps_hugging_face_clients_offline():
    # Even an explicit HF_HUB_OFFLINE=0 on another machine must not let core setup reach the Hub.
    assert "HUB_OFFLINE True True" in hub_mode(HF_HUB_OFFLINE="0", HF_DATASETS_OFFLINE="0")


def test_celeba_preparation_may_use_the_hub():
    assert "HUB_OFFLINE False False" in hub_mode("--celeba")
