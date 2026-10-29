"""What raw EEG-IO looks like.

The dataset ships no figure of its own and its ``read_data.py`` is Python 2,
so this draws one:

  A  the file exactly as written -- no filtering, no baseline removal, so the
     multi-mV electrode offset and the drift are visible
  B  the same trace detrended and band-passed 0.5-10 Hz, with the hand
     annotations and any corrupt interval drawn over it
  C  four consecutive blinks on both electrodes, showing the artifact is
     common-mode rather than neural
  D  the blink-locked median over every usable blink in all 20 subjects

Panels A-C are one recording (S00 by default: it has both a visible drift and
an annotated corrupt interval).  Panel D pools all subjects.

    python figures/eegio_raw.py                      # -> figures/eegio_raw.png
    python figures/eegio_raw.py --subject S13 --out /tmp/s13.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, detrend

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Searched in order, as ``gui/eegio_tab.py`` does: the dataset may sit loose in
#: ``data/`` or still inside the ``EEG-EyeBlinks/`` bundle it is distributed in.
DATA_DIR_CANDIDATES = (
    os.path.join(ROOT, "data", "EEG-IO"),
    os.path.join(ROOT, "data", "EEG-EyeBlinks", "EEG-IO"),
)


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    return DATA_DIR_CANDIDATES[0]


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--data_dir", default=None, help="EEG-IO directory (both layouts are found)")
ap.add_argument("--subject", default="S00", help="recording drawn in panels A-C")
ap.add_argument("--out", default=os.path.join(HERE, "eegio_raw.png"))
args = ap.parse_args()

D = args.data_dir or find_data_dir()
OUT = args.out
FS = 250.0

if not os.path.isdir(D):
    sys.exit(f"No EEG-IO recordings in {D!r}.\n"
             f"  Searched: {', '.join(DATA_DIR_CANDIDATES)}\n"
             f"  EEG-IO is not redistributable; see data/EEGIO.py.")

# --- design tokens (validated reference palette, light surface) ---
SURF, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CODE = {0: ("#2a78d6", "-",  "normal blink (code 0)"),
        1: ("#eb6834", "-",  "stimulated blink (code 1)"),
        2: ("#1baf7a", ":",  "soft blink (code 2)")}
CRIT = "#d03b3b"

def load(s):
    return np.loadtxt(os.path.join(D, s + "_data.csv"), delimiter=";", skiprows=1, usecols=(0, 1, 2))

def labels(s):
    corr, bl, n = [], [], 0
    for row in open(os.path.join(D, s + "_labels.csv")):
        r = [c.strip() for c in row.split(",")]
        if r[0] == "corrupt":
            n = int(r[1])
        elif n > 0:
            corr.append((float(r[0]), float(r[1]))); n -= 1
        elif r[0] == "blinks":
            pass
        elif len(r) >= 2:
            bl.append((float(r[0]), int(r[1])))
    return corr, np.array(bl)

B, A = butter(2, [0.5, 10], btype="band", fs=FS)
clean = lambda x: filtfilt(B, A, detrend(x))

def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for k, sp in ax.spines.items():
        sp.set_visible(k in ("left", "bottom")); sp.set_color(BASE); sp.set_lw(1.0)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(SEC)

SUB = args.subject
a = load(SUB); corr, bl = labels(SUB)
t, fp1, fp2 = a[:, 0], a[:, 1], a[:, 2]

fig = plt.figure(figsize=(13.5, 10.2), facecolor=SURF)
gs = GridSpec(4, 3, figure=fig, hspace=0.62, wspace=0.28,
              left=0.075, right=0.985, top=0.925, bottom=0.065)

# ---- A: exactly what is in the file ----
axA = fig.add_subplot(gs[0, :]); style(axA)
axA.plot(t, fp1, lw=0.9, color="#2a78d6", label="Fp1")
axA.plot(t, fp2, lw=0.9, color="#eb6834", label="Fp2")
axA.axhline(0, color=BASE, lw=1.0, ls="--", zorder=1)
axA.set_ylabel("potential  (µV, as stored)", fontsize=9, color=SEC)
axA.set_title("A   The file as written — no filtering, no baseline removal.  Each channel sits on its own multi-mV electrode offset and drifts.",
              fontsize=10.5, color=INK, loc="left", pad=8)
axA.legend(frameon=False, fontsize=8.5, labelcolor=SEC, ncol=2, loc="upper right")
axA.annotate(f"Fp1 offset ≈ {fp1.mean():+,.0f} µV", xy=(t[-1]*0.012, fp1.mean()),
             fontsize=8.5, color=SEC, va="bottom")
axA.annotate(f"Fp2 offset ≈ {fp2.mean():+,.0f} µV", xy=(t[-1]*0.012, fp2.mean()),
             fontsize=8.5, color=SEC, va="top")
axA.set_xlim(0, t[-1])

# ---- B: the same trace, usable, with the annotations ----
axB = fig.add_subplot(gs[1, :]); style(axB)
x1 = clean(fp1)
axB.plot(t, x1, lw=0.85, color=INK, zorder=3)
for c0, c1 in corr:
    axB.axvspan(c0, c1, color=CRIT, alpha=0.22, zorder=1, lw=0)
seen = set()
for bt, bc in bl:
    col, ls, lab = CODE[int(bc)]
    axB.axvline(bt, color=col, ls=ls, lw=1.1, alpha=0.9, zorder=2)
    seen.add(int(bc))
axB.set_ylabel("Fp1  (µV)", fontsize=9, color=SEC)
axB.set_xlabel("time  (s)", fontsize=9, color=SEC)
axB.set_title("B   Detrended and band-passed 0.5–10 Hz — the blinks are the only thing left.  Vertical rules are the hand-annotated instants; the band is a corrupt interval.",
              fontsize=10.5, color=INK, loc="left", pad=8)
hs = [Line2D([], [], color=CODE[c][0], ls=CODE[c][1], lw=1.4, label=CODE[c][2]) for c in sorted(seen)]
hs.append(Patch(facecolor=CRIT, alpha=0.22, label="corrupt interval"))
axB.legend(handles=hs, frameon=False, fontsize=8.5, labelcolor=SEC, ncol=4, loc="upper right")
axB.set_xlim(0, t[-1])
ok = np.ones(len(x1), bool)
for c0, c1 in corr:
    ok[(t >= c0) & (t <= c1)] = False
lim = 1.9 * np.percentile(np.abs(x1[ok]), 99.5)
axB.set_ylim(-lim, lim)
if np.abs(x1[~ok]).max() > lim:
    axB.annotate(f"excursion inside the corrupt band reaches {np.abs(x1[~ok]).max():,.0f} µV — clipped here",
                 xy=(corr[0][1], -lim), xytext=(8, 8), textcoords="offset points",
                 fontsize=7.5, color=CRIT, va="bottom", ha="left")

# ---- C: zoom on three blinks, both channels ----
i0 = int(np.searchsorted(bl[:, 0], 20.0))
z0, z1 = bl[i0, 0] - 3.0, bl[i0 + 3, 0] + 3.0
m = (t >= z0) & (t <= z1)
axC = fig.add_subplot(gs[2:, :2]); style(axC)
axC.plot(t[m], clean(fp1)[m], lw=1.5, color="#2a78d6", label="Fp1")
axC.plot(t[m], clean(fp2)[m], lw=1.5, color="#eb6834", label="Fp2")
first = True
for bt, bc in bl:
    if z0 <= bt <= z1:
        axC.axvline(bt, color=MUTED, ls="--", lw=1.0, zorder=2)
        if first:
            axC.annotate("annotated instant", xy=(bt, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -10), textcoords="offset points", fontsize=7.5,
                         color=MUTED, va="top", ha="left")
            first = False
axC.set_xlabel("time  (s)", fontsize=9, color=SEC)
axC.set_ylabel("µV", fontsize=9, color=SEC)
axC.set_title("C   Four consecutive blinks, both electrodes.  Near-identical\non Fp1 and Fp2 — the artifact is common-mode, not neural.",
              fontsize=10.5, color=INK, loc="left", pad=8)
axC.legend(frameon=False, fontsize=8.5, labelcolor=SEC, ncol=2, loc="lower right")
axC.set_xlim(z0, z1)

# ---- D: blink-locked median over all 20 subjects ----
axD = fig.add_subplot(gs[2:, 2]); style(axD)
half = int(0.6 * FS)
segs = []
for i in range(20):
    s = f"S{i:02d}"
    aa = load(s); _, bb = labels(s)
    xx = clean(aa[:, 1])
    for bt, bc in bb:
        j = int(bt * FS)
        if j - half >= 0 and j + half < len(xx):
            segs.append(xx[j - half:j + half])
segs = np.array(segs)
tt = (np.arange(-half, half)) / FS
med = np.median(segs, axis=0)
q1, q3 = np.percentile(segs, 25, axis=0), np.percentile(segs, 75, axis=0)
axD.fill_between(tt, q1, q3, color="#2a78d6", alpha=0.20, lw=0)
axD.plot(tt, med, lw=2.0, color="#2a78d6")
axD.axvline(0, color=MUTED, ls="--", lw=1.0)
axD.set_xlabel("time from annotation  (s)", fontsize=9, color=SEC)
axD.set_ylabel("µV", fontsize=9, color=SEC)
axD.set_title(f"D   Blink shape: median of\n{len(segs):,} blinks, IQR band",
              fontsize=10.5, color=INK, loc="left", pad=8)
axD.set_xlim(tt[0], tt[-1])

fig.suptitle("EEG-IO — 20 subjects, Fp1/Fp2, 250 Hz, ~102 s each, involuntary blinks annotated by hand",
             fontsize=12.5, color=INK, x=0.075, ha="left", y=0.978)
fig.savefig(OUT, dpi=170, facecolor=SURF, bbox_inches="tight")
print("wrote", OUT)
print(f"panel D: {len(segs)} blink segments; median peak-to-peak {np.ptp(med):.0f} uV")
