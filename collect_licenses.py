"""Copy exact installed dependency notices into a Windows distribution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parent
DISTRIBUTIONS = (
    "mido", "numpy", "soundfile", "cffi", "pycparser", "packaging",
    "pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile",
    "pywin32-ctypes", "setuptools", "typing_extensions",
)


def collect_licenses(destination: Path) -> dict:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inventory = {"python": sys.version.split()[0], "dependencies": {}, "notices": []}

    def copy(source, relative):
        source = Path(source)
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError("License destination escaped its output directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory["notices"].append({"path": target.relative_to(destination).as_posix(),
                                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    for name in DISTRIBUTIONS:
        distribution = metadata.distribution(name)
        inventory["dependencies"][name] = distribution.version
        count = 0
        for entry in distribution.files or ():
            path = PurePosixPath(str(entry).replace("\\", "/"))
            if ".." in path.parts or path.is_absolute():
                continue
            lower = path.name.lower()
            is_notice = any(word in lower for word in ("license", "copying", "copyright", "notice"))
            is_notice = is_notice or "licensing" in path.parts
            if not is_notice or path.suffix.lower() in (".py", ".pyc", ".pyd", ".dll"):
                continue
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                copy(source, Path(name) / str(path))
                count += 1
        if not count:
            raise RuntimeError(f"No installed license notice found for {name}.")

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("The Python installation must include LICENSE.txt.")
    copy(python_license, Path("Python") / "LICENSE.txt")

    # Some Python installers omit Tcl's source license. The upstream license
    # bundled with this source release supplies that notice without network IO.
    import tkinter
    tcl_library = Path(tkinter.Tcl().eval("info library"))
    tcl_notice = tcl_library / "license.terms"
    if not tcl_notice.is_file():
        tcl_notice = ROOT / "licenses" / "upstream" / "Tcl-license.terms"
    copy(tcl_notice, Path("Tcl") / "license.terms")
    tk_notices = sorted(tcl_library.parent.glob("tk*/license.terms"))
    if not tk_notices:
        tk_notices = sorted((Path(sys.base_prefix) / "tcl").glob("tk*/license.terms"))
    if not tk_notices:
        raise RuntimeError("The Python/Tk installation must include Tk's license.terms.")
    copy(tk_notices[-1], Path("Tk") / "license.terms")
    for source in sorted((ROOT / "licenses" / "upstream").glob("*")):
        if source.is_file():
            copy(source, Path("upstream") / source.name)
    inventory["notices"].sort(key=lambda row: row["path"])
    (destination / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = collect_licenses(args.destination)
    print(f"Collected {len(result['notices'])} dependency notices.")
