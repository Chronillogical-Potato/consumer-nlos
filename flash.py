"""Flash the VL53L8CH firmware onto a Nucleo-F401RE board.

Vendored from cc-hardware:
  cc_hardware/tools/flash.py             (vl53l8ch_flash)
  cc_hardware/utils/misc/serial_utils.py (find_device_by_label)

The firmware source tree lives in ``cleaned/firmware/vl53l8ch/``. Building
and uploading is done by the existing makefile under that tree.

Usage::

    python flash.py             # auto-detect mounted board, build + upload
    python flash.py /Volumes/NOD_F401RE
    python flash.py --no-build  # skip the make step (re-upload existing .bin)

Prereqs: an ARM bare-metal toolchain (``arm-none-eabi-gcc`` & friends) and
``make`` on PATH. On macOS::

    brew install --cask gcc-arm-embedded
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parent / "firmware" / "vl53l8ch"
BUILD_DIR = FIRMWARE_DIR / "build"
NUCLEO_LABEL = "NOD_F401RE"


def find_device_by_label(label: str) -> Path | None:
    """Return the mount path of an attached drive whose name contains ``label``.

    Stdlib-only: walks ``/Volumes`` on macOS and ``/media/<user>`` / ``/mnt`` on
    Linux. The Nucleo's bootloader presents itself as a FAT volume named
    ``NOD_F401RE``.
    """
    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates.append(Path("/Volumes"))
    else:
        user = os.environ.get("USER", "")
        if user:
            candidates.append(Path(f"/media/{user}"))
        candidates.extend([Path("/media"), Path("/mnt")])

    for root in candidates:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if label in entry.name:
                return entry
    return None


def flash(
    port: Path | None = None, *, build: bool = True, verbose: bool = True
) -> None:
    if port is None:
        port = find_device_by_label(NUCLEO_LABEL)
        if port is None:
            raise RuntimeError(
                f"Could not find {NUCLEO_LABEL} volume. Plug the board in via USB "
                "(it should mount as a drive named NOD_F401RE) and try again."
            )

    port = Path(port)
    if not port.exists():
        raise RuntimeError(f"Port {port} does not exist")

    if not BUILD_DIR.exists():
        raise RuntimeError(f"Firmware build dir not found at {BUILD_DIR}")

    quiet = "" if verbose else "-s"

    if build:
        print(f"Building firmware in {BUILD_DIR}")
        if os.system(f"make -C {BUILD_DIR} clean all {quiet}") != 0:
            raise RuntimeError("Build failed")

    print(f"Uploading firmware from {BUILD_DIR} to {port}")
    if os.system(f"make -C {BUILD_DIR} upload PORT={port}") != 0:
        raise RuntimeError("Upload failed")

    print("Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Flash VL53L8CH firmware.")
    parser.add_argument(
        "port",
        nargs="?",
        default=None,
        help="Mount path of the Nucleo (e.g. /Volumes/NOD_F401RE). "
        "Auto-detected by USB label if omitted.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip `make clean all` and re-upload the existing .bin.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress make output (pass -s to make).",
    )
    args = parser.parse_args()

    try:
        flash(
            port=Path(args.port) if args.port else None,
            build=not args.no_build,
            verbose=not args.quiet,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
