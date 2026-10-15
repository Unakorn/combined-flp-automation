"""Headless checks for portable setup and GUI-to-runner configuration."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import combined_gui as gui
import combined_settings as settings


class Variable:
    def __init__(self, value=None, **kwargs):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class PublicGuiTests(unittest.TestCase):
    def setUp(self):
        # mkdir inherits ACLs on Windows, including restricted build/test hosts.
        self.folder = Path(tempfile.gettempdir()) / ("oa-public-gui-" + uuid.uuid4().hex)
        self.folder.mkdir()
        self.source = self.folder / "stems.flp"
        self.source.write_bytes(b"input")
        self.midi = self.folder / "song.mid"
        self.midi.write_bytes(b"input")
        self.template = self.folder / "effects.json"
        self.template.write_text("{}", encoding="utf-8")
        self.base = dict(source=str(self.source), output=str(self.folder / "new.flp"), midi="", original="",
                         eq_calibration="", section_template="", modules=dict(volume=True, section=False, eq=False),
                         section_enabled=dict(filter=False, delay=False, width=False, pan=True, reverb=False),
                         max_cut_db="1.5", volume_intensity="Medium")

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_fresh_ui_has_no_project_defaults_and_volume_only(self):
        with patch.object(gui, "load_settings", return_value={}), patch.object(gui, "template_path", return_value=self.folder / "none.json"), \
             patch.object(gui.tk, "StringVar", Variable), patch.object(gui.tk, "BooleanVar", Variable), \
             patch.object(gui.CombinedAutomationApp, "_build"):
            app = gui.CombinedAutomationApp(Mock())
        self.assertEqual([app.source.get(), app.midi.get(), app.original.get(), app.output.get(), app.eq_calibration.get()], [""] * 5)
        self.assertTrue(app.volume.get())
        self.assertFalse(app.section.get())
        self.assertFalse(app.eq.get())
        self.assertEqual({key: var.get() for key, var in app.section_enabled.items()},
                         dict(filter=False, delay=False, width=False, pan=True))

    def test_volume_only_ignores_inactive_midi_and_eq_settings(self):
        config = deepcopy(self.base)
        config.update(midi="missing.mid", original="missing.flp", eq_calibration="missing.flp", max_cut_db="invalid")
        result = gui.validate_config(config)
        self.assertEqual(result["midi"], "")
        self.assertEqual(result["original"], "")
        self.assertEqual(result["max_cut_db"], 1.5)

    def test_native_pan_requires_midi_but_no_effect_template(self):
        config = deepcopy(self.base)
        config["modules"]["section"] = True
        with self.assertRaisesRegex(ValueError, "full-song midi"):
            gui.validate_config(deepcopy(config))
        config["midi"] = str(self.midi)
        with patch.object(gui, "effect_templates", side_effect=AssertionError("No template needed")):
            self.assertEqual(gui.validate_config(config)["section_template"], "")

    def test_enabled_effect_requires_matching_imported_family(self):
        config = deepcopy(self.base)
        config["modules"]["section"] = True
        config["section_enabled"]["filter"] = True
        config["midi"] = str(self.midi)
        with self.assertRaisesRegex(ValueError, "section effects setup"):
            gui.validate_config(deepcopy(config))
        config["section_template"] = str(self.template)
        importer = Mock()
        importer.load_template.return_value = {"families": {"delay": {}}}
        with patch.object(gui, "effect_templates", return_value=importer), self.assertRaisesRegex(ValueError, "filter"):
            gui.validate_config(deepcopy(config))
        importer.load_template.return_value = {"families": {"filter": {}}}
        with patch.object(gui, "effect_templates", return_value=importer):
            self.assertEqual(gui.validate_config(config)["section_template"], str(self.template.resolve()))

    def test_disabled_section_choices_cannot_generate_empty_stage(self):
        config = deepcopy(self.base)
        config["modules"]["section"] = True
        config["midi"] = str(self.midi)
        config["section_enabled"] = dict.fromkeys(config["section_enabled"], False)
        with self.assertRaisesRegex(ValueError, "at least one section"):
            gui.validate_config(config)

    def test_existing_output_cannot_be_replaced(self):
        config = deepcopy(self.base)
        Path(config["output"]).write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            gui.validate_config(config)
        self.assertEqual(Path(config["output"]).read_bytes(), b"keep")

    def test_eq_requires_user_calibration(self):
        config = deepcopy(self.base)
        config["modules"]["eq"] = True
        with self.assertRaisesRegex(ValueError, "eq setup flp"):
            gui.validate_config(config)

    def test_setup_persists_only_paths_and_effect_choices(self):
        with patch.object(settings, "settings_directory", return_value=self.folder):
            settings.save_settings("my-effects.json", "my-calibration.flp", {"filter": True, "pan": True})
            result = settings.load_settings()
        self.assertEqual(result["section_template"], "my-effects.json")
        self.assertEqual(result["eq_calibration"], "my-calibration.flp")
        self.assertTrue(result["section_enabled"]["filter"])
        stored = json.loads((self.folder / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(set(stored), {"version", "section_template", "eq_calibration", "section_enabled"})

    def test_corrupt_or_foreign_settings_are_ignored(self):
        with patch.object(settings, "settings_directory", return_value=self.folder):
            for payload in ("garbage", "[]", '{"version":2}', '{"version":1,"section_template":false}'):
                (self.folder / "settings.json").write_text(payload, encoding="utf-8")
                self.assertEqual(settings.load_settings(), {})

    def test_import_uses_private_user_folder_and_enables_imported_effects(self):
        app = gui.CombinedAutomationApp.__new__(gui.CombinedAutomationApp)
        app.setup_window = Mock()
        app.section_template = Variable("")
        app.eq_calibration = Variable("")
        app.section_enabled = {key: Variable(key == "pan") for key, _ in gui.SECTION_FAMILIES}
        app.setup_status = Variable("")
        app._remember_setup = Mock()
        importer = Mock()
        importer.import_template.return_value = {"path": str(self.template), "families": {
            "filter": {"ready": True}, "delay": {"ready": False}, "width": {"ready": True}}}
        with patch.object(gui.filedialog, "askopenfilename", return_value=str(self.source)), \
             patch.object(gui, "effect_templates", return_value=importer), \
             patch.object(gui, "template_path", return_value=self.template):
            app._import_setup()
        importer.import_template.assert_called_once_with(str(self.source), out_path=self.template)
        self.assertTrue(app.section_enabled["filter"].get())
        self.assertTrue(app.section_enabled["width"].get())
        self.assertFalse(app.section_enabled["delay"].get())
        app._remember_setup.assert_called_once()

    def test_worker_delivers_success_and_failure_without_widgets(self):
        events = queue.Queue()
        backend = Mock()
        backend.run_combined.return_value = {"output": "new.flp"}
        with patch.object(gui.importlib, "import_module", return_value=backend):
            gui.execute_generation({"source": "stems.flp"}, events)
        self.assertEqual(events.get_nowait(), ("complete", {"output": "new.flp"}))
        backend.run_combined.side_effect = ValueError("Check setup")
        with patch.object(gui.importlib, "import_module", return_value=backend):
            gui.execute_generation({}, events)
        event, payload = events.get_nowait()
        self.assertEqual(event, "error")
        self.assertEqual(payload["message"], "Check setup")
        self.assertIn("ValueError", payload["traceback"])

    def test_missing_configuration_stops_before_worker_starts(self):
        app = gui.CombinedAutomationApp.__new__(gui.CombinedAutomationApp)
        app.running = False
        app.root = Mock()
        app._configuration = Mock(side_effect=ValueError("Choose a stem FLP file."))
        with patch.object(gui.messagebox, "showerror") as showerror, \
             patch.object(gui.threading, "Thread") as worker:
            app._generate()
        worker.assert_not_called()
        showerror.assert_called_once_with("Check project settings", "Choose a stem FLP file.", parent=app.root)
        self.assertFalse(app.running)

    def event_app(self):
        app = gui.CombinedAutomationApp.__new__(gui.CombinedAutomationApp)
        app.root = Mock()
        app.events = queue.Queue()
        app.status = Variable("")
        app.output = Variable("fallback.flp")
        app._append = Mock()
        app._set_running = Mock()
        return app

    def test_pending_worker_error_is_displayed_and_unlocks_controls(self):
        app = self.event_app()
        app.events.put(("error", {"message": "Invalid effects setup", "traceback": "ValueError: Invalid effects setup"}))
        with patch.object(gui.messagebox, "showerror") as showerror:
            app._poll_events()
        app._set_running.assert_called_once_with(False)
        showerror.assert_called_once_with("Automation could not be generated", "Invalid effects setup", parent=app.root)
        self.assertIn("Generation failed", app.status.get())
        self.assertIn("Invalid effects setup", app._append.call_args.args[0])
        app.root.after.assert_called_once_with(100, app._poll_events)

    def test_pending_success_displays_saved_files_and_unlocks_controls(self):
        app = self.event_app()
        app.events.put(("complete", {"output": "new.flp", "report_path": "report.json", "summary": "2 editable clips"}))
        app._poll_events()
        app._set_running.assert_called_once_with(False)
        log = "\n".join(call.args[0] for call in app._append.call_args_list)
        self.assertIn("new.flp", log)
        self.assertIn("report.json", log)
        self.assertIn("2 editable clips", log)
        self.assertIn("Saved for listening review", app.status.get())


if __name__ == "__main__":
    unittest.main()
