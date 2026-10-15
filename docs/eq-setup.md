# EQ setup

EQ currently supports **FabFilter Pro-Q 3.29, Windows x64 VST3**, file version
3.2.9.62. You must install and license it yourself. The program verifies the
plugin binary and the parameter mapping before exporting. Other plugin builds
are rejected until their mapping is tested; the other automation modules can
still be used with EQ switched off.

The supported binary SHA-256 is
`bdb70fc219cc62d507d79d81b60e5ab986dd0afb356a86a978c1df9ca1827430`.
The installation folder is unrestricted. Your saved FLP must reference the
actual local `.vst3` file. If the project came from another computer, locate
Pro-Q in FL Studio and save a local copy first.

## Make your own calibration FLP

Use a small separate project that contains one Pro-Q instance on a dedicated
mixer insert numbered 1–63. This project is a local setup file; it is not
uploaded anywhere.

1. Prepare enabled Bell bands **3, 4, 5, 6 and 7**, each with **Q = 1.000**,
   **Stereo** placement, **dynamic range = 0**, and Solo disabled. Suggested
   frequencies are 76, 254, 806, 2460 and 7519 Hz respectively. The program
   reads your actual saved frequencies.
2. Start each of those five gains at **0 dB**. Bands 1 and 2 may exist; they
   are switched off in newly inserted copies. Bands 8–24 must be unused.
   Creating seven bands first is a convenient way to obtain band numbers 3–7.
3. Use neutral global settings: output level and pan zero, gain scale at its
   normal 100%, processing mode Zero Latency, phase invert off, Auto Gain off,
   MIDI input off, no solo or external side-chain/speaker processing, and no
   plugin/host bypass. Leave the FL effect slot enabled at **100% mix**.
4. Create a normal FL Studio Automation Clip linked to **Band 7 Gain** (or
   another gain among bands 3–7). Give it a name containing `Band 7 Gain` for
   that band. Use the default identity link and full clip Min/Max range;
   disable the clip's LFO. Place the enabled clip once in the Playlist. A flat
   curve is sufficient. Do not create duplicate gain clips for the same band.
5. Save the FLP and choose it in the app's **EQ calibration FLP** field.

All gain clips in the calibration should target the same Pro-Q instance so
the app can select it automatically. The calibration does not need your song,
audio stems or MIDI. Its native automation link verifies the FL parameter
address used by the output clips.

## What happens to the stem project

When no stem has Pro-Q, the program copies only the calibrated plugin record
into a free slot on each dedicated stem insert. It prepares zero-gain bands
3–7, disables bands 1–2, and checks all 24 bands and processing globals through
the installed plugin controller before writing the new FLP. It does not copy
calibration channels, notes, audio or Playlist clips into the song.

When every stem already has exactly one Pro-Q instance, the app preserves
those saved plugin states and adds supported gain automation relative to each
saved gain value. Its calibration link still verifies the address mapping.
Mixed projects with missing or duplicate Pro-Q instances are rejected with
instructions to make the setup consistent.

EQ produces small, cut-only gain changes based on the actual stem audio.
Only eligible, unowned controls receive automation. A band with an unsupported
shape, Q, dynamic range, placement or existing automation is skipped. The
output report lists skipped controls and validation results. Open the new FLP
in FL Studio and review the result before using it in a mix.

No FabFilter binary, personal calibration project or song audio is included in
the source or executable. The executable includes Python; it does not replace
FL Studio or the required audio plugins.
