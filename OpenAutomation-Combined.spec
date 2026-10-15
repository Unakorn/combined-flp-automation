# PyInstaller one-folder distribution: keep runtime DLLs replaceable.
from pathlib import Path

ROOT = Path(SPECPATH)
RESOURCES = (
    "automation-effects.json", "stem-effects.json", "stem-names.json",
    "stem-native.json", "stem-presets.json",
)
UNUSED = {"eq_gui", "section_comparisons", "section_gui", "section_matching",
          "section_profiles", "stem_gui"}
backend = sorted(p.stem for p in (ROOT / "work").glob("*.py") if p.stem not in UNUSED)
hidden = ["app_paths", "combined_runner", "combined_gui", "combined_inputs",
          "combined_eq_setup", "combined_settings", "packaging_smoke"] + backend

a = Analysis(
    [str(ROOT / "Combined Automation.py")],
    pathex=[str(ROOT), str(ROOT / "work")],
    binaries=[],
    datas=[(str(ROOT / "resources" / name), "resources") for name in RESOURCES],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OpenAutomation-Combined",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    hide_console="hide-early",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False,
    name="OpenAutomation-Combined",
)
