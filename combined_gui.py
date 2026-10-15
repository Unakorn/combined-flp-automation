"""Single-window launcher for the combined FLP automation pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from combined_settings import load_settings, save_settings, settings_directory, template_path


VOLUME_AMOUNTS = ("Low", "Medium-Low", "Medium", "Medium-High", "High")
SECTION_FAMILIES = (("filter", "F6 filter"), ("delay", "H-Delay"),
                    ("width", "Auto Shape Stereo width"), ("pan", "Native mixer pan"))
SETUP_HELP = """START WITH VOLUME
Choose a stem FLP and a new output filename. Volume automation works without MIDI or extra effects. Stems must be continuous from song start, routed to separate Mixer inserts and named by instrument (for example Lead, Pad, Arp, Bass, Kick, Drums or Vocal).

SECTION EFFECTS AND PAN
Supply a full-song MIDI export with track names matching your stems. Native pan needs no effect setup. Enable only the effects you have configured in this window.

1. In a small FL Studio project, load your installed effects in Mixer slots. Keep each enabled with its FL slot Mix at 100%. Set its other knobs as you want them saved.
2. Create a native Automation Clip linked directly to each desired knob. Name it exactly:
   OA Filter — Waves F6 low-pass frequency
   OA Delay — Waves H-Delay Dry/Wet
   OA Reverb — Waves RVerb Wet/Dry
   OA Width — Auto Shape Stereo Width
3. Keep clip Min/Max at the full 0–1 range and LFO off. Use exactly four points: saved baseline → safe lower bound → safe upper bound → saved baseline. Use one direct knob link per clip.
4. Filter starts at its upper/open bound. Delay and reverb start at their lower/dry bound. Width starts between its bounds. Set each knob back to baseline, use FL's “Init song with this position”, then save the FLP.
5. Click “Import effects setup FLP…”. Repeat to add missing effects. Your imported setup is saved locally; your FLP and audio are not copied. Use the same installed plugin components for playback.

PRO-Q 3 EQ
Save a small FLP containing the supported Pro-Q 3.29 x64 VST3 with active Bell bands 3–7, Q=1, Stereo placement and dynamic range=0. Other bands must be unused, except bands 1/2 which the generated flat copy will disable. Turn off solo and keep global output gain/pan neutral, auto gain off and gain scale at unity. Create a native Automation Clip directly linked to a Band 3–7 Gain knob and keep its standard descriptive name (for example “Pro-Q 3 - Band 7 Gain”). Choose this FLP as EQ setup. The linked instance is detected automatically.

The EQ module verifies the installed, supported Pro-Q 3 VST3 build before writing. When your stems have no Pro-Q instances, it adds a verified flat copy. If they already have exactly one supported instance each, their saved settings are preserved. Incompatible setups stop with an explanation.

FL Studio and any chosen effect plugins must be installed separately. The executable includes Python. Open the newly saved FLP in FL Studio to listen and adjust the editable clips."""


def effect_templates():
    """Load the shared importer in both a source checkout and a frozen build."""
    work = str(Path(__file__).resolve().parent / "work")
    if work not in sys.path:
        sys.path.insert(0, work)
    return importlib.import_module("plugin_templates")


def next_output_path(source: str) -> str:
    """Choose a new sibling output without replacing a previous generation."""
    if not source.strip():
        return ""
    path = Path(source)
    destination = path.with_name(f"{path.stem} - Combined Automation.flp")
    sequence = 2
    while destination.exists():
        destination = path.with_name(
            f"{path.stem} - Combined Automation {sequence}.flp"
        )
        sequence += 1
    return str(destination)


def validate_config(config: dict) -> dict:
    """Validate user input before the worker touches an FLP."""
    modules = config["modules"]
    if not any(modules.values()):
        raise ValueError("Select at least one automation module.")
    required = {"source": ("Stem FLP", {".flp"})}
    if modules["section"]:
        required["midi"] = ("Full-song MIDI", {".mid", ".midi"})
    if modules["eq"]:
        required["eq_calibration"] = ("EQ setup FLP", {".flp"})
    optional = {"original": ("Original instrument FLP", {".flp"})} if modules["section"] else {}
    if not modules["section"]:
        config["midi"] = config["original"] = ""
    selected = config.get("section_enabled", {"pan": True})
    if modules["section"]:
        if not any(selected.values()):
            raise ValueError("Enable at least one section effect or native pan in Effects setup.")
        plugin_families = [family for family in ("filter", "delay", "width", "reverb") if selected.get(family)]
        if plugin_families:
            required["section_template"] = ("Section effects setup", {".json"})
    for key, (label, suffixes) in {**optional, **required}.items():
        value = str(config.get(key, "")).strip()
        if not value:
            if key in required:
                raise ValueError(f"Choose a {label.lower()} file.")
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"{label} was not found:\n{path}")
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"{label} must be a {' or '.join(sorted(suffixes))} file.")
        config[key] = str(path.resolve())
    if modules["section"] and plugin_families:
        template = effect_templates().load_template(config["section_template"])
        missing = [family for family in plugin_families if family not in template["families"]]
        if missing:
            raise ValueError("Import a setup for these enabled effects, or turn them off: " + ", ".join(missing))

    output_value = str(config.get("output", "")).strip()
    if not output_value:
        raise ValueError("Choose a filename for the new combined FLP.")
    output = Path(output_value).expanduser()
    if output.suffix.lower() != ".flp":
        raise ValueError("The output filename must end in .flp.")
    if output.exists():
        raise ValueError("That output already exists. Choose a new filename.")
    if not output.parent.is_dir():
        raise ValueError("Choose an existing output folder.")
    if output.resolve() in {
        Path(config[key]).resolve()
        for key in ("source", "original", "eq_calibration")
        if config.get(key)
    }:
        raise ValueError("The output must be a new file, separate from your input projects.")
    config["output"] = str(output.resolve())

    try:
        maximum = float(config["max_cut_db"]) if modules["eq"] else 1.5
    except (TypeError, ValueError):
        raise ValueError("Maximum EQ cut must be a number between 0.1 and 3 dB.") from None
    if not 0.1 <= maximum <= 3:
        raise ValueError("Maximum EQ cut must be between 0.1 and 3 dB.")
    config["max_cut_db"] = maximum
    if modules["volume"] and config["volume_intensity"] not in VOLUME_AMOUNTS:
        raise ValueError("Choose a volume amount from the list.")
    return config


def execute_generation(config: dict, events: queue.Queue) -> None:
    """Worker entry point: never access Tk widgets from this thread."""
    def progress(*parts, **details):
        message = " ".join(str(part) for part in parts)
        if not message and details:
            message = str(details.get("message", details))
        if message:
            events.put(("progress", message))

    try:
        backend = importlib.import_module("combined_runner")
        result = backend.run_combined(config, progress=progress)
        events.put(("complete", result))
    except Exception as exc:
        events.put(("error", {"message": str(exc), "traceback": traceback.format_exc()}))


class CombinedAutomationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Open Automation — Combined FLP")
        self.root.geometry("900x760")
        self.root.minsize(750, 680)
        self.events: queue.Queue = queue.Queue()
        self.running = False
        self.setup_window = None
        self.editable_widgets: list[tuple[tk.Widget, str]] = []
        saved = load_settings()
        self.source = tk.StringVar(value="")
        self.midi = tk.StringVar(value="")
        self.original = tk.StringVar(value="")
        self.output = tk.StringVar(value="")
        self.eq_calibration = tk.StringVar(value=saved.get("eq_calibration", ""))
        self.section_template = tk.StringVar(value=saved.get("section_template", "") or
                                             (str(template_path()) if template_path().is_file() else ""))
        choices = saved.get("section_enabled", {})
        self.section_enabled = {key: tk.BooleanVar(value=choices.get(key, key == "pan"))
                                for key, _ in SECTION_FAMILIES}
        self.volume = tk.BooleanVar(value=True)
        self.section = tk.BooleanVar(value=False)
        self.eq = tk.BooleanVar(value=False)
        self.add_reverb = tk.BooleanVar(value=False)
        self.volume_intensity = tk.StringVar(value="Medium")
        self.max_cut_db = tk.StringVar(value="1.5")
        self.status = tk.StringVar(value="Ready. Generate a new FLP, then audition it in FL Studio.")
        self.setup_status = tk.StringVar(value="Volume is ready. Configure optional effects once in Effects setup.")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)

    def _editable(self, widget: tk.Widget, state: str = "normal") -> tk.Widget:
        self.editable_widgets.append((widget, state))
        return widget

    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(7, weight=1)
        ttk.Label(frame, text="Combined FLP automation", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Add volume movement, section effects and dynamic EQ in one pass.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 16))

        files = ttk.LabelFrame(frame, text="Project files", padding=12)
        files.grid(row=2, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self._file_row(files, 0, "Stem FLP", self.source, "flp")
        self._file_row(files, 1, "Full-song MIDI", self.midi, "midi")
        self._file_row(files, 2, "Original FLP (optional)", self.original, "flp")
        self._file_row(files, 3, "Save new FLP as", self.output, "output")

        modules = ttk.LabelFrame(frame, text="Automation", padding=12)
        modules.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        modules.columnconfigure(2, weight=1)
        self._editable(ttk.Checkbutton(modules, text="Volume balance, builds and fades", variable=self.volume)).grid(
            row=0, column=0, sticky="w", padx=(0, 16), pady=4
        )
        ttk.Label(modules, text="Amount").grid(row=0, column=1, sticky="w", padx=(0, 8))
        self._editable(ttk.Combobox(modules, textvariable=self.volume_intensity, values=VOLUME_AMOUNTS,
                                   state="readonly", width=15), "readonly").grid(row=0, column=2, sticky="w")
        self._editable(ttk.Checkbutton(modules, text="Section effects and pan (MIDI required)", variable=self.section)).grid(
            row=1, column=0, sticky="w", padx=(0, 16), pady=4
        )
        self._editable(ttk.Checkbutton(modules, text="Add reverb", variable=self.add_reverb)).grid(
            row=1, column=1, columnspan=2, sticky="w"
        )
        self._editable(ttk.Checkbutton(modules, text="Audio-driven Pro-Q 3 EQ", variable=self.eq)).grid(
            row=2, column=0, sticky="w", padx=(0, 16), pady=4
        )
        ttk.Label(modules, text="Max cut (dB)").grid(row=2, column=1, sticky="w", padx=(0, 8))
        self._editable(ttk.Spinbox(modules, textvariable=self.max_cut_db, from_=0.1, to=3.0,
                                  increment=0.1, width=7)).grid(row=2, column=2, sticky="w")
        ttk.Label(modules, text="Leave Add reverb off when reverb is already rendered into the stems.").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        setup = ttk.Frame(frame)
        setup.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        setup.columnconfigure(1, weight=1)
        self._editable(ttk.Button(setup, text="Effects setup…", command=self._open_setup)).grid(row=0, column=0, sticky="w")
        ttk.Label(setup, textvariable=self.setup_status, wraplength=650).grid(row=0, column=1, sticky="w", padx=(12, 0))

        action = ttk.Frame(frame)
        action.grid(row=5, column=0, sticky="ew", pady=(16, 8))
        action.columnconfigure(1, weight=1)
        self.generate_button = ttk.Button(action, text="Generate combined FLP", command=self._generate)
        self.generate_button.grid(row=0, column=0, sticky="w")
        self.busy = ttk.Progressbar(action, mode="indeterminate", length=180)
        self.busy.grid(row=0, column=1, sticky="e")
        ttk.Label(frame, textvariable=self.status, wraplength=820).grid(row=6, column=0, sticky="ew", pady=(0, 8))

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=7, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=8, wrap="word", state="disabled", font=("Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        ttk.Label(frame, text="Your source projects are preserved. Existing output files cannot be replaced.").grid(
            row=8, column=0, sticky="w", pady=(8, 0)
        )

    def _file_row(self, parent, row, label, variable, kind, *, track=True):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        entry = ttk.Entry(parent, textvariable=variable)
        button = ttk.Button(parent, text="Browse…", command=lambda: self._browse(variable, kind))
        if track:
            self._editable(entry)
            self._editable(button)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        button.grid(row=row, column=2, padx=(8, 0), pady=5)

    def _browse(self, variable, kind):
        old_value = variable.get()
        current = Path(old_value) if old_value else None
        initial_directory = current.parent if current else Path(self.source.get()).parent
        if not initial_directory.is_dir():
            initial_directory = Path.cwd()
        options = {"parent": self.root, "initialdir": str(initial_directory)}
        if kind == "output":
            path = filedialog.asksaveasfilename(
                **options, title="Save a new combined FLP", defaultextension=".flp",
                initialfile=current.name if current else "Combined Automation.flp",
                filetypes=[("FL Studio project", "*.flp")],
            )
        else:
            filetypes = {"midi": [("MIDI file", "*.mid *.midi")],
                         "json": [("Effects setup", "*.json")]}.get(kind, [("FL Studio project", "*.flp")])
            path = filedialog.askopenfilename(**options, title="Choose project input", filetypes=filetypes)
        if path:
            variable.set(path)
            if variable is self.source:
                self.output.set(next_output_path(path))

    def _open_setup(self):
        if self.running:
            return
        if self.setup_window is not None and self.setup_window.winfo_exists():
            self.setup_window.lift()
            return
        window = self.setup_window = tk.Toplevel(self.root)
        window.title("Optional effects setup")
        window.geometry("860x720")
        window.minsize(730, 550)
        window.transient(self.root)
        window.grab_set()
        panel = ttk.Frame(window, padding=16)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(7, weight=1)
        self._file_row(panel, 0, "Section setup JSON", self.section_template, "json", track=False)
        ttk.Button(panel, text="Import effects setup FLP…", command=self._import_setup).grid(row=1, column=1, sticky="w", pady=(3, 8))
        choices = ttk.Frame(panel)
        choices.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 10))
        for index, (key, label) in enumerate(SECTION_FAMILIES):
            ttk.Checkbutton(choices, text=label, variable=self.section_enabled[key]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 22), pady=3)
        self._file_row(panel, 3, "EQ setup FLP", self.eq_calibration, "flp", track=False)
        ttk.Label(panel, text="Setup paths are remembered on this computer. Section and EQ can be enabled in the main window.",
                  wraplength=800).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(panel, textvariable=self.setup_status, wraplength=800).grid(row=5, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Label(panel, text="Setup guide", font=("Segoe UI", 11, "bold")).grid(row=6, column=0, sticky="w", pady=(8, 5))
        help_frame = ttk.Frame(panel)
        help_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        help_frame.columnconfigure(0, weight=1)
        help_frame.rowconfigure(0, weight=1)
        body = tk.Text(help_frame, wrap="word", height=14, font=("Segoe UI", 10), padx=8, pady=8)
        body.insert("1.0", SETUP_HELP)
        body.configure(state="disabled")
        body.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(help_frame, command=body.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        body.configure(yscrollcommand=scroll.set)
        ttk.Button(panel, text="Save setup and close", command=self._close_setup).grid(row=8, column=2, sticky="e", pady=(12, 0))
        window.protocol("WM_DELETE_WINDOW", self._close_setup)

    def _remember_setup(self):
        save_settings(self.section_template.get().strip(), self.eq_calibration.get().strip(),
                      {key: variable.get() for key, variable in self.section_enabled.items()})

    def _close_setup(self):
        try:
            self._remember_setup()
        except OSError as exc:
            messagebox.showerror("Could not save setup", str(exc), parent=self.setup_window)
            return
        effects = [label for key, label in SECTION_FAMILIES if self.section_enabled[key].get()]
        self.setup_status.set("Selected: " + (", ".join(effects) or "no section effects") +
                              (". EQ setup selected." if self.eq_calibration.get().strip() else "."))
        self.setup_window.destroy()
        self.setup_window = None

    def _import_setup(self):
        chosen = filedialog.askopenfilename(parent=self.setup_window, title="Import your FL-saved effects setup",
                                           filetypes=[("FL Studio project", "*.flp")])
        if not chosen:
            return
        try:
            result = effect_templates().import_template(chosen, out_path=template_path())
            self.section_template.set(result["path"])
            ready = [key for key, item in result["families"].items() if item["ready"]]
            for key in ready:
                if key in self.section_enabled:
                    self.section_enabled[key].set(True)
            self._remember_setup()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Effects setup could not be imported", str(exc), parent=self.setup_window)
            return
        self.setup_status.set("Imported: " + ", ".join(ready) + ". Saved in " + str(settings_directory()))

    def _configuration(self) -> dict:
        return validate_config({
            "source": self.source.get().strip(),
            "midi": self.midi.get().strip(),
            "original": self.original.get().strip(),
            "output": self.output.get().strip(),
            "eq_calibration": self.eq_calibration.get().strip(),
            "section_template": self.section_template.get().strip(),
            "section_enabled": {**{key: value.get() for key, value in self.section_enabled.items()},
                                "reverb": self.add_reverb.get()},
            "modules": {"volume": self.volume.get(), "section": self.section.get(), "eq": self.eq.get()},
            "max_cut_db": self.max_cut_db.get(),
            "volume_intensity": self.volume_intensity.get(),
            "seed": 150,
            "reverb_mode": "add" if self.add_reverb.get() else "baked",
            "section_intensities": {key: "auto" for key in ("filter", "delay", "width", "pan")},
        })

    def _generate(self):
        if self.running:
            return
        try:
            config = self._configuration()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Check project settings", str(exc), parent=self.root)
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._set_running(True)
        self.status.set("Generating automation. Progress will appear below.")
        self._append(f"Source: {config['source']}\nOutput: {config['output']}")
        threading.Thread(target=execute_generation, args=(config, self.events), daemon=False).start()

    def _set_running(self, active: bool):
        self.running = active
        self.generate_button.configure(state="disabled" if active else "normal")
        for widget, normal_state in self.editable_widgets:
            widget.configure(state="disabled" if active else normal_state)
        if active:
            self.busy.start(12)
        else:
            self.busy.stop()
            self.busy.configure(value=0)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text).rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_events(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    self._append(payload)
                elif event == "complete":
                    self._set_running(False)
                    self.status.set("Saved for listening review. Open the combined FLP in FL Studio and audition it.")
                    self._append(f"\nSaved FLP: {payload.get('output', self.output.get())}")
                    if payload.get("report_path"):
                        self._append(f"Report: {payload['report_path']}")
                    if payload.get("summary"):
                        self._append(payload["summary"])
                    self._append("Native FL Studio playback has not been verified; audition the saved project before using it.")
                elif event == "error":
                    self._set_running(False)
                    self.status.set("Generation failed. Review the message below, adjust settings, then try again.")
                    self._append(f"\nERROR: {payload['message']}\n\n{payload['traceback']}")
                    messagebox.showerror("Automation could not be generated", payload["message"], parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _close(self):
        if self.running:
            messagebox.showinfo("Generation in progress", "Please wait for generation to finish before closing.", parent=self.root)
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CombinedAutomationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
