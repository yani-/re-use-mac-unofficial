"""Download the mamba-ssm source release that the tests compare against.

The source is only unpacked, never built or installed (it needs CUDA). The
tests load two pure-Python files from it: selective_scan_ref and the Mamba
module's non-fused forward path.
"""

import hashlib
import io
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "2.3.2.post1"
URL = f"https://files.pythonhosted.org/packages/source/m/mamba-ssm/mamba_ssm-{VERSION}.tar.gz"
SHA256 = "104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028"
DEST = Path(__file__).resolve().parent.parent / ".mamba_ref"


def main():
    if (DEST / f"mamba_ssm-{VERSION}").is_dir():
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {URL}")
    try:  # python.org builds on macOS ship without CA certs (certifi: requirements-dev.txt)
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    data = urllib.request.urlopen(URL, context=ctx).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        sys.exit(f"SHA-256 mismatch: expected {SHA256}, got {digest}")
    DEST.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(DEST, filter="data")
    print(f"Unpacked to {DEST}")


if __name__ == "__main__":
    main()
