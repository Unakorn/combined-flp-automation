# Open Automation — Combined FLP Automation

Create editable FL Studio automation for a rendered-stem project in one pass: **volume balance and builds**, optional **section effects and pan**, and optional **FabFilter Pro-Q 3 band gain**. Everything runs locally; no account, AI service, API key, or internet connection is needed to generate automation.

## Download and run without Python

1. Download **OpenAutomation-Combined-1.0.0-windows-x64.zip** from the [release page](https://github.com/Unakorn/combined-flp-automation/releases/tag/v1.0.0).
2. Extract the entire ZIP to a folder you can write to.
3. Open **OpenAutomation-Combined.exe**. Keep its `_internal` folder beside it.

The Windows x64 package includes Python, Tcl/Tk and the audio/MIDI libraries. It does not include FL Studio, commercial plugins, or any music. FL Studio and the effects you select must be installed separately. The build is unsigned.

## First song

1. In FL Studio, save a **stem-only FLP** with one continuous WAV clip per stem, all beginning at the song start, including silence. Use a separate mixer insert per stem, constant tempo, 4/4, and one Playlist arrangement. Consolidate edits such as stretching, trimming, pitch changes and reversal before using the app.
2. Name stems by their instrument: for example **Lead, Pad, Arp, Bass, Drums, Kick, Vocal**. Keep the referenced WAVs available. The app reads those files; it does not render stems.
3. Choose the stem FLP and a new output filename. Leave **Volume** checked for a first run. This mode needs neither MIDI nor additional effect plugins.
4. Click **Generate combined FLP**. Open the new project in FL Studio, listen, and adjust the native Automation Clips as you like.

The original files are preserved, existing output filenames are refused, and a JSON report is saved beside the new FLP. The report contains local input paths and hashes; review it before sharing it. Audio stays in its existing location and is not copied into the output FLP.

## Enable everything in one run

Check **Volume**, **Section effects**, and **EQ** after completing the setup below. Each module can also run independently. Volume runs last, and the section engine leaves volume to that module so two engines do not compete for the same knob.

- **Volume:** audio-driven balance, smooth builds and fades using native mixer volume. It keeps the saved mix as its reference; this is not mastering or a target-loudness processor.
- **Section effects and pan:** supply a matching **full-song multitrack MIDI export**, including arrangement repeats and gaps. Name MIDI tracks to match stems. Optional original instrument FLP verifies notes and routing; it does not replace MIDI. A Pad→Chords alias requires matching routing in that original FLP. Shared drum MIDI supplies section timing, not individual kick-note identification. Native pan works without plugin setup. Choose the effect families in **Effects setup** and import your own calibration for the effects you enable.
- **EQ:** requires the supported **FabFilter Pro-Q 3 VST3 3.29** binary and an FL-saved native gain-link setup. See [EQ setup](docs/eq-setup.md). If there are no Pro-Q instances on the stem inserts, the app adds verified flat instances. If each stem already has exactly one compatible instance, it preserves their saved states. Mixed or duplicate setups need adjustment first.

Section effects support **Waves F6 low-pass frequency**, **H-Delay Dry/Wet**, **RVerb Wet/Dry**, and **Auto Shape Stereo Width**. Your installed versions and saved wrappers must be compatible with your own calibration. Reverb is off by default because stems may already contain reverb. Follow [Effects setup](EFFECTS_SETUP.md) or the guide in the setup window.

EQ analyzes the original rendered audio and static saved faders; it does not simulate the later effects or volume automation. Unknown roles, conflicting automation, unsupported project layouts or plugin versions are reported or rejected rather than assigned guessed links. This is an experimental FLP reader/writer, not an Image-Line or plugin-vendor product. Structural validation does not replace listening in FL Studio.

## Editable source

Download **OpenAutomation-Combined-1.0.0-source.zip**, or clone [this repository](https://github.com/Unakorn/combined-flp-automation). The application is in the repository root. The complete Python source, resource definitions, tests, and Windows build recipe are included under the [MIT license](LICENSE). Third-party packages keep their own licenses.

Use Windows x64 with Python 3.12 and Tcl/Tk:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe "Combined Automation.py"
```

For command-line use, both the EXE and Python entry point accept `--config path.json`, `--version`, and `--self-test`. Start from [example-config.json](example-config.json), replace its example paths, and select the modules you want. Volume-only is the default. Files referenced by the config must exist; the output must be new.

## Build the Windows download

Run from the repository root or the extracted source folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.lock
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe build_windows.py --version 1.0.0 --output-dir dist
```

The build produces an EXE folder, a Windows ZIP, a source ZIP and SHA-256 checksums. Build on Windows x64; PyInstaller does not cross-compile this application. See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled runtime licenses and replaceable shared libraries. The repository's [Windows build workflow](.github/workflows/windows-build.yml) uses the same recipe on pushes to `main` or manual runs. It uploads build artifacts; publishing a release is a separate action.

Setup choices and imported section calibration are stored under `%LOCALAPPDATA%\OpenAutomation`. Song selections are not remembered. Intermediate runs and analysis caches are stored under the system temporary directory in `OpenAutomationCombined`; they are not uploaded by the app.
