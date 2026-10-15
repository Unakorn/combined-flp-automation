"""Headless checks of the packaged runtime; never opens FL Studio or Tk windows."""
from __future__ import annotations

from io import BytesIO
import json
import math
import sys
import tempfile
from pathlib import Path
from uuid import uuid4


def self_test() -> dict:
    from app_paths import bundle_root, resource_path
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(bundle_root() / "work"))
    import numpy as np
    import soundfile as sf
    import mido
    import tkinter
    import combined_runner
    import combined_gui
    import combined_eq_setup
    from stem_native import build_channel, parse_points
    from stem_names import NAME_RULES_PATH
    from stem_planner import load_presets, PRESET_PATH
    from stem_effects import CATALOG as STEM_CATALOG
    from flp_connector import CATALOG as CONNECTOR_CATALOG
    from plugin_templates import DEFAULT_TEMPLATE
    from combined_settings import template_path

    resource_names = ("automation-effects.json", "stem-effects.json", "stem-names.json",
                      "stem-native.json", "stem-presets.json")
    for name in resource_names:
        with resource_path(name).open(encoding="utf-8") as source:
            assert isinstance(json.load(source), dict), name
    assert NAME_RULES_PATH == resource_path("stem-names.json")
    assert PRESET_PATH == resource_path("stem-presets.json")
    assert STEM_CATALOG == resource_path("stem-effects.json")
    assert CONNECTOR_CATALOG == resource_path("automation-effects.json")
    assert DEFAULT_TEMPLATE == template_path()
    assert not resource_path("local-plugin-template.json").exists()
    assert load_presets()

    samples = (np.sin(np.arange(512, dtype=np.float64) * (2 * math.pi / 64)) * .125).astype(np.float32)
    waveform = BytesIO()
    sf.write(waveform, samples, 8000, format="WAV", subtype="FLOAT")
    waveform.seek(0)
    restored, rate = sf.read(waveform, dtype="float32")
    assert rate == 8000 and np.array_equal(samples, restored)
    assert np.isfinite(np.fft.rfft(restored)).all()

    midi = mido.MidiFile(ticks_per_beat=96)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_off", note=60, time=96),
    ]))
    payload = BytesIO()
    midi.save(file=payload)
    payload.seek(0)
    assert mido.MidiFile(file=payload).ticks_per_beat == 96

    points = [(0, .5), (96, .25), (384, .5)]
    channel = build_channel(0, "Self test", points, 96)
    assert parse_points(dict(channel)[234], 96) == points
    tcl_version = tkinter.Tcl().eval("info patchlevel")
    # Both dispatcher paths must return a bounded JSON error through file IPC.
    # A deliberately absent VST path ensures that no real plugin is loaded.
    from eq_vst3_probe import _isolated_probe, PINNED_PLUGIN_SHA256
    missing_plugin = Path(tempfile.gettempdir()) / ("oa-missing-plugin-" + uuid4().hex + ".vst3")
    request = {"plugin_path": str(missing_plugin), "plugin_sha256": PINNED_PLUGIN_SHA256}
    for flag in ("--probe-child", "--probe-full-child"):
        try:
            _isolated_probe(request, flag, bundle_root() / "Combined Automation.py", timeout=20)
        except ValueError as exc:
            if "isolated process" not in str(exc) or "unavailable" not in str(exc):
                raise AssertionError("Unexpected probe worker failure: " + str(exc)) from exc
        else:
            raise AssertionError("A probe worker accepted a missing plugin.")
    return {
        "ok": True, "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
        "tcl": tcl_version,
        "checks": ["engine imports", "resource locations", "local template path",
                   "WAV write/read", "FFT", "MIDI write/read", "native curve roundtrip", "Tcl runtime",
                   "EQ worker error IPC", "full EQ worker error IPC"],
        "native_fl_playback_tested": False,
    }
