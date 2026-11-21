"""
Paired EEG-IO test: TCN+SE (L=0) versus TCN+MLA r16, one saved test per seed.

Same settings as the Pi runs of the MLA models: causal 1-80 Hz filter + 60 Hz notch, input as
recorded, hop 4, 20 participants, 250 Hz. TCN+SE is listed first, so its output sets the
per-participant threshold for the adjusted blink count, and both models are counted with it.
Runs through gui.eegio_tab.evaluate (the EEG-IO tab's own entry point), so each test is saved in
eegio_tests/ and appears in the tab's history.

    ../env/bin/python paper_results/eegio_mla_vs_se.py 67 72 88
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui import eegio_tab  # noqa: E402


def main(seeds):
    made = {}
    for seed in seeds:
        runs = [f"tcn_se_both_250hz_fixed_s{seed}", f"tcn_mla_r16_both_250hz_fixed_s{seed}"]
        print("seed", seed, runs, flush=True)
        out = eegio_tab.evaluate({
            "runs": runs, "n_subjects": 20, "hop": 4, "channel": 0, "target_fs": 250.0,
            "zero_phase": False, "rescale_input": False, "adjusted_blink_count": True,
        })
        made[seed] = out["name"]
        print("  saved", out["name"], flush=True)
    (ROOT / "paper_results" / "eegio_mla_vs_se_tests.json").write_text(json.dumps(made, indent=1))
    print("done", made)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [67, 72, 88])
