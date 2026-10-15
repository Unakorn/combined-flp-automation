"""Build and smoke-test a Windows x64 folder ZIP and an editable source ZIP.

Run with Python 3.12 after installing requirements-build.lock.
Generated files are confined to build/ and the selected output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

from collect_licenses import collect_licenses

ROOT = Path(__file__).resolve().parent
APP_NAME = "OpenAutomation-Combined"
RESOURCES = ("automation-effects.json", "stem-effects.json", "stem-names.json",
             "stem-native.json", "stem-presets.json")
UNUSED = {"eq_gui", "section_comparisons", "section_gui", "section_matching",
          "section_profiles", "stem_gui"}
ROOT_SUFFIXES = {".py", ".md", ".txt", ".lock", ".spec", ".ps1", ".bat"}
TEST_DATA = {"gentle_fx_unaffected_golden.json", "fx_controls_legacy_golden.json"}


def public_source_files() -> list[Path]:
    """Explicit content boundary; never recurse into caches or generated work."""
    paths = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix.lower() in ROOT_SUFFIXES or p.name in {"LICENSE", "VERSION", ".gitignore", "example-config.json"})]
    paths += [p for p in (ROOT / "work").glob("*.py") if p.stem not in UNUSED]
    paths += [ROOT / "resources" / name for name in RESOURCES]
    paths += [p for p in (ROOT / "tests").glob("*") if p.is_file() and
              (p.suffix == ".py" or p.name in TEST_DATA)]
    paths += [p for p in (ROOT / "licenses" / "upstream").glob("*") if p.is_file()]
    paths += [p for p in (ROOT / "docs").glob("*.md") if p.is_file()]
    paths += [ROOT / ".github" / "workflows" / "windows-build.yml"]
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("The source archive may not follow links outside the source tree.")
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())


def add_zip_file(archive, source: Path, name: str):
    # Stable metadata for source archives; file contents remain byte-for-byte.
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source.read_bytes())


def build(output_dir: Path, version: str) -> dict:
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Build the Windows release on Windows x64.")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for the recorded release environment.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("Version must use letters, numbers, dots, underscores or hyphens.")
    for required in ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md"):
        if not (ROOT / required).is_file():
            raise FileNotFoundError(f"Required release document: {required}")
    source_files = public_source_files()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app_dir = (output_dir / APP_NAME).resolve()
    if not app_dir.is_relative_to(output_dir) or app_dir == output_dir:
        raise ValueError("Invalid generated application directory.")
    # PyInstaller may replace only this generated child directory. It must not
    # overlap the source tree or any of its ancestors.
    if ROOT.is_relative_to(app_dir):
        raise ValueError("The output directory would overwrite application source.")
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(build_dir / "pyinstaller-cache")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm",
               "--distpath", str(output_dir), "--workpath", str(build_dir / "pyinstaller"),
               str(ROOT / "OpenAutomation-Combined.spec")]
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    inventory = collect_licenses(app_dir / "licenses")
    for name in ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "EFFECTS_SETUP.md",
                 "example-config.json", "RELEASE_NOTES.md"):
        shutil.copyfile(ROOT / name, app_dir / name)
    (app_dir / "docs").mkdir(exist_ok=True)
    for path in (ROOT / "docs").glob("*.md"):
        shutil.copyfile(path, app_dir / "docs" / path.name)
    workflow_dir = app_dir / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / ".github" / "workflows" / "windows-build.yml",
                    workflow_dir / "windows-build.yml")
    executable = app_dir / (APP_NAME + ".exe")
    # No desktop UI is opened; the bootloader keeps redirected pipes available.
    result = subprocess.run([str(executable), "--self-test"], cwd=output_dir,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError("Frozen self-test failed:\n" + result.stdout + result.stderr)
    smoke = json.loads(result.stdout)
    if not smoke.get("ok") or not smoke.get("frozen"):
        raise RuntimeError("The executable did not confirm its frozen runtime self-test.")
    info = {"version": version, "platform": "Windows x64", "python": sys.version.split()[0],
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": inventory["dependencies"], "self_test": smoke,
            "source_files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files}}
    (app_dir / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    windows_zip = output_dir / f"{APP_NAME}-{version}-windows-x64.zip"
    source_zip = output_dir / f"{APP_NAME}-{version}-source.zip"
    with zipfile.ZipFile(windows_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                add_zip_file(archive, path, APP_NAME + "/" + path.relative_to(app_dir).as_posix())
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files:
            add_zip_file(archive, path, APP_NAME + "-source/" + path.relative_to(ROOT).as_posix())
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
                            for p in (windows_zip, source_zip)), encoding="utf-8")
    return {"folder": str(app_dir), "windows_zip": str(windows_zip),
            "source_zip": str(source_zip), "checksums": str(sums), "self_test": smoke}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.version), indent=2))


if __name__ == "__main__":
    main()
