# Section effect setup

Volume and native pan require no effect calibration. Configure only the plugin families you want to use. The app copies your FL-saved plugin state and a real parameter link; it does not guess plugin parameter addresses.

1. Create a small setup FLP. Insert your effects on mixer inserts 1–63, enabled, with FL's slot Mix at 100%.
2. Create a native Automation Clip for each desired **internal plugin control**. Use these exact clip names:

   | Clip name | Plugin and control |
   | --- | --- |
   | `OA Filter` | Waves F6 low-pass frequency |
   | `OA Delay` | Waves H-Delay Dry/Wet |
   | `OA Reverb` | Waves RVerb Wet/Dry |
   | `OA Width` | Auto Shape Stereo Width |

3. Use a direct identity link to one control per clip. Keep Min/Max at the full 0–1 range and LFO off. Create exactly four straight-line points at increasing times: **baseline → lower bound → upper bound → baseline**. Pick bounds by reading the plugin's displayed values, not the physical extremes of its knob.
4. Filter baseline equals its upper/open bound. Delay/reverb baseline equals its lower/dry bound. Width baseline lies between its bounds. Set the actual knob back to baseline, choose FL's **Init song with this position**, and save.
5. Open **Effects setup → Import effects setup FLP…** in the combined app. You can import all effects together or add families later. Turn on only the imported families you intend to use.

Imported settings are stored locally under `%LOCALAPPDATA%\OpenAutomation`. They are specific to your installed plugin components. Do not include a personal setup in a public fork unless you have reviewed its contents.

Suggested starting states:

- F6: enable its low-pass with a gentle slope; leave other gains flat, dynamic ranges zero, unwanted high-pass off, and output 0 dB. Baseline is open. Choose a restrained lower cutoff for your music.
- H-Delay: Dry/Wet 0, Output 0 dB, Analog Off, modulation Depth 0, a fixed host-synced division and modest feedback. Keep delay time and character fixed; automate only the wet amount.
- RVerb: Wet/Dry 0, a restrained room/plate and nonnegative predelay. Keep time, type, size and decay fixed. Added reverb is off by default in the combined app.
- Auto Shape Stereo: reset the shape, select HELD, set all eight band widths to 1, and manually set every Generated control to 0. Set global Width to 100%, Stereoize and Low-end Mono off, Output 0 dB and Mix 100%. Reset shape alone does not reset all of these settings.

These settings are starting points, not a claim that every saved plugin state is neutral. Import checks the link, identity, bounds and baseline structure. Audition the setup in FL Studio. Wet-control fades also affect existing delay/reverb tails; this version does not isolate notes through auxiliary sends.
