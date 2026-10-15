The complete combined FLP automation app is now available as editable MIT-licensed Python source and a portable Windows x64 download.

- **Windows users:** download `OpenAutomation-Combined-1.0.0-windows-x64.zip`, extract everything, and run `OpenAutomation-Combined.exe`. Python installation is not required. Keep the `_internal` folder with the EXE.
- **Source users:** download `OpenAutomation-Combined-1.0.0-source.zip`, or clone the repository. The app lives at the repository root. Source, tests, resource definitions and build scripts are included.
- **Checksums:** `SHA256SUMS.txt` lists the SHA-256 hashes of both ZIP files.

Volume balance/builds, section effects/pan, and Pro-Q 3 gain automation can be selected together in one run. Start with Volume; use Effects setup to configure optional plugin modules using your own FL-saved calibration. Native volume needs no MIDI or effect plugins. Section effects need matching full-song MIDI. EQ supports the verified Pro-Q 3 VST3 3.29 build.

FL Studio and selected commercial effect plugins are separate requirements. No plugins, songs, stems, MIDI, personal project reports or private calibration files are included. The app creates a new FLP with editable clips and a local generation report, preserving the source project.

The Windows package is unsigned. Automated checks cover the bundled runtime, worker processes, native links, preservation, matching and export behavior. Review the generated project by listening in FL Studio; compatibility with every FL/plugin version is not claimed.

See the [guide](https://github.com/Unakorn/combined-flp-automation#readme) for setup, source installation and rebuilding.
