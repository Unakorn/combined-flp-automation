"""Start the combined window, or run a JSON configuration with --config."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'work'))
VERSION = '1.0.0'


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('--probe-child', '--probe-full-child'):
        from eq_vst3_probe import worker_main, _probe_child
        if sys.argv[1] == '--probe-full-child':
            from combined_eq_setup import _probe_child_full
            return worker_main(_probe_child_full)
        return worker_main(_probe_child)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=f'Open Automation Combined {VERSION}')
    parser.add_argument('--self-test', action='store_true', help='Check the packaged runtime without opening a window.')
    parser.add_argument('--config', help='Run a saved JSON configuration without opening a window.')
    args = parser.parse_args()
    if args.self_test:
        from packaging_smoke import self_test
        print(json.dumps(self_test(), indent=2))
    elif args.config:
        from combined_runner import run_combined
        config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        result = run_combined(config, progress=lambda message: print(message, flush=True))
        print(json.dumps({k: result[k] for k in ('output', 'report_path', 'summary')}, indent=2))
    else:
        from combined_gui import main as open_window
        open_window()


if __name__ == "__main__":
    raise SystemExit(main())
