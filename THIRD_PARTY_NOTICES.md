# Third-party runtime notices

The Windows ZIP includes a Python runtime and open-source dependencies so the app can run without installing Python. Their original notices are in the adjacent `licenses/` directory. `licenses/inventory.json` records the installed versions and notice checksums used for that build.

| Component | License / source |
| --- | --- |
| Python | [PSF and included licenses](https://docs.python.org/3.12/license.html) |
| Tcl/Tk | [Tcl/Tk license](https://www.tcl-lang.org/software/tcltk/license.html) |
| Mido | [MIT](https://github.com/mido/mido) |
| NumPy and included numerical libraries | [BSD and component notices](https://github.com/numpy/numpy) |
| SoundFile | [BSD 3-Clause](https://github.com/bastibe/python-soundfile/tree/0.14.0) |
| libsndfile | [LGPL 2.1 or later; version 1.2.2 source](https://github.com/libsndfile/libsndfile/tree/1.2.2) |
| CFFI / pycparser | [CFFI](https://github.com/python-cffi/cffi) / [pycparser](https://github.com/eliben/pycparser) licenses |
| packaging | [Apache 2.0 / BSD](https://github.com/pypa/packaging) |
| PyInstaller | [GPL with bootloader exception; runtime hooks Apache 2.0](https://pyinstaller.org/en/stable/license.html) |

SoundFile's Windows library contains additional codecs. The packaged `licenses/soundfile/licensing/license_notes.md` provides their copyright notices and source links. The exact dependency notices, including the NumPy wheel's bundled-library notices and SoundFile's LGPL text, are retained rather than replaced by this summary. Build-tool notices are included as well.

This is a folder distribution. Keep the executable and `_internal` directory together. The libsndfile DLL remains a separate library under `_internal/_soundfile_data/` and can be replaced with a compatible rebuilt version. The source distribution and build script let you rebuild the application with modified dependencies. No restriction is imposed on modifying those libraries or reverse engineering the application to debug those modifications.

FL Studio, FabFilter Pro-Q 3, Waves F6/H-Delay/RVerb, Auto Shape Stereo, and other music plugins are not included. Their names belong to their respective owners. Native FL serialization definitions and effect settings are interoperability data; this project does not grant a license to third-party software, factory content, or trademarks. Users supply their own installed plugins and local calibration files.

The executable is built with PyInstaller's one-folder mode. The build configuration and runtime resource lookup follow the [PyInstaller spec-file documentation](https://pyinstaller.org/en/stable/spec-files.html) and [runtime information documentation](https://pyinstaller.org/en/stable/runtime-information.html).
