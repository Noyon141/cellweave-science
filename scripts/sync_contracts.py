"""Deterministic Contract Synchronization Tool.

Enforces that cellweave-science consumes contract models generated strictly
from cellweave-contracts, guaranteeing a single source of truth and zero schema drift.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_contracts_repo_root() -> Path:
    return get_repo_root().parent / "cellweave-contracts"


def calculate_dir_sha256(directory: Path) -> str:
    """Calculate deterministic SHA256 of all python files in directory."""
    hasher = hashlib.sha256()
    for root, _, files in sorted(os.walk(directory)):
        for f in sorted(files):
            if f.endswith(".py"):
                path = Path(root) / f
                hasher.update(path.read_bytes())
    return hasher.hexdigest()


def sync_contracts(check_only: bool = False) -> int:
    science_root = get_repo_root()
    contracts_root = get_contracts_repo_root()
    vendor_dir = science_root / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    wheel_dest = vendor_dir / "cellweave_contracts-0.1.0-py3-none-any.whl"

    if not contracts_root.exists():
        if wheel_dest.exists():
            print(
                f"Contracts source not present at {contracts_root}. Using locked vendor wheel {wheel_dest.name}."
            )
            return 0
        print(
            f"ERROR: Contracts repo not found at {contracts_root} and no vendor wheel at {wheel_dest}.",
            file=sys.stderr,
        )
        return 1

    generated_py_dir = contracts_root / "generated" / "python" / "cellweave_contracts"
    if not generated_py_dir.exists():
        print(
            f"ERROR: Generated contracts directory missing at {generated_py_dir}.", file=sys.stderr
        )
        return 1

    current_sha256 = calculate_dir_sha256(generated_py_dir)
    hash_file = vendor_dir / ".contracts_sha256"

    if check_only:
        if not wheel_dest.exists():
            print(
                f"ERROR: Vendor wheel {wheel_dest} does not exist. Run sync_contracts without --check.",
                file=sys.stderr,
            )
            return 1
        if hash_file.exists() and hash_file.read_text().strip() != current_sha256:
            print(
                "ERROR: Schema drift detected! cellweave-contracts has changed. Run 'python scripts/sync_contracts.py' to update.",
                file=sys.stderr,
            )
            return 1
        print(f"Contract authority check passed. Source hash: {current_sha256[:12]}...")
        return 0

    print(f"Building cellweave-contracts wheel from {contracts_root}...")
    temp_build = vendor_dir / "_build"
    temp_build.mkdir(exist_ok=True)

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                str(contracts_root),
                "-w",
                str(temp_build),
            ],
            stdout=subprocess.PIPE,
        )
        built_wheels = list(temp_build.glob("cellweave_contracts*.whl"))
        if not built_wheels:
            print("ERROR: Failed to build wheel from contracts repo.", file=sys.stderr)
            return 1
        shutil.copy2(built_wheels[0], wheel_dest)
        hash_file.write_text(current_sha256)
        print(
            f"Successfully synchronized contract wheel to {wheel_dest} (hash {current_sha256[:12]}...)."
        )
    finally:
        shutil.rmtree(temp_build, ignore_errors=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize contracts from cellweave-contracts")
    parser.add_argument("--check", action="store_true", help="Check for drift without rebuilding")
    args = parser.parse_args()
    sys.exit(sync_contracts(check_only=args.check))
