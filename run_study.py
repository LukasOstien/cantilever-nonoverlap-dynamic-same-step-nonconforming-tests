#!/usr/bin/env python3
"""
Sweep robin_parameter x overlap x h_free/h_clamped ratio for the cantilever
Schwarz-impedance-nonoverlap problem.

For every combination:
  1. rewrite the top-of-file APREPRO variables in cantilever-nonconforming.jou
  2. rewrite `robin parameter:` in cantilever-clamped.yaml + cantilever-free.yaml
  3. regenerate meshes:   julia journal-to-exodus.jl cantilever-nonconforming.jou
  4. run the simulation:  norma cantilever-multi.yaml
  5. concatenate the per-domain total-energy CSVs, remove the raw ones
  6. compute drift(t) = (E(t)/E(0))*100 - 100 and keep it in memory

For each (robin, overlap) pair, all mesh ratios are plotted together and the
figure is written to  robin_<robin>/overlap_<pct>pct.png .

Total cantilever length is fixed at 0.254 m; overlap is symmetric, so the two
subdomains have equal length (length + overlap) / 2.

The original .jou / .yaml files are restored on exit (including on Ctrl-C).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Study parameters -- edit here.
# ---------------------------------------------------------------------------
ROBIN_PARAMS      = [1.0e9,2.0e9,5.0e9,8.0e9,1.0e11,2.0e11,5.0e11,8.0e11,1.0e12,2.0e12,5.0e12,8.0e12]
OVERLAP_FRACTIONS = [0.0]
MESH_RATIOS       = [1/3, 0.5, 0.6, 0.75, 0.8, 1.0]        # h_free = ratio * h_clamped
# MESH_RATIOS       = [1.0]  
H_CLAMPED         = 0.008467
CANTILEVER_LENGTH = 0.254
DT_PLOT           = 1.0e-6                  # spacing between energy samples

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
JOU          = ROOT / "cantilever-nonconforming.jou"
YAML_CLAMPED = ROOT / "cantilever-clamped.yaml"
YAML_FREE    = ROOT / "cantilever-free.yaml"
YAML_MULTI   = ROOT / "cantilever-multi.yaml"


# ---------------------------------------------------------------------------
# File rewriters
# ---------------------------------------------------------------------------
def rewrite_jou(*, h_clamped: float, h_free: float,
                domain_length_clamped: float, domain_length_free: float,
                length: float) -> None:
    """Replace the value inside each `${name = ...}` at the top of the .jou."""
    text = JOU.read_text()
    values = {
        "h_clamped":             h_clamped,
        "h_free":                h_free,
        "domain_length_clamped": domain_length_clamped,
        "domain_length_free":    domain_length_free,
        "length":                length,
    }
    for name, val in values.items():
        pattern = rf"(\$\{{\s*{re.escape(name)}\s*=\s*)[^}}]+(\}})"
        text, n = re.subn(pattern, lambda m, v=val: f"{m.group(1)}{v!r}{m.group(2)}",
                          text, count=1)
        if n == 0:
            raise RuntimeError(f"variable ${{{name} = ...}} not found in {JOU.name}")
    JOU.write_text(text)


def rewrite_robin(yaml_path: Path, robin: float) -> None:
    """Replace `robin parameter: <number>` (one occurrence per yaml)."""
    text = yaml_path.read_text()
    new_text, n = re.subn(r"(robin parameter:\s*)[0-9.eE+\-]+",
                          lambda m: f"{m.group(1)}{robin:.6e}", text)
    if n == 0:
        raise RuntimeError(f"'robin parameter:' not found in {yaml_path.name}")
    if n > 1:
        raise RuntimeError(f"expected one 'robin parameter:' line in {yaml_path.name}, found {n}")
    yaml_path.write_text(new_text)


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------
def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_shell(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, cwd=ROOT, shell=True, check=True)


def load_total_energy(csv: Path) -> np.ndarray:
    """
    Read the `total` column from norma's blended-energy CSV. The file has a
    header row followed by rows of  time, stored, kinetic, total  (columns are
    comma-separated), so we skip row 0 and take column index 3.
    """
    arr = np.loadtxt(csv, delimiter=",", skiprows=1)
    return arr[:, 3]


def cleanup_stage_outputs() -> None:
    """Wipe intermediates before the next inner iteration."""
    for f in ROOT.glob("cantilever*.csv"):
        f.unlink()


def save_overlap_plot(out: Path, curves: list[tuple[str, np.ndarray]],
                      robin: float, frac: float) -> None:
    """(Re)render the (robin, overlap) figure with all mesh-ratio curves collected so far."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, drift in curves:
        t = np.arange(len(drift)) * DT_PLOT
        ax.plot(t, drift, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total energy drift (%)")
    ax.set_title(f"robin = {robin:.1e},  overlap = {int(frac*100)}%")
    # scientific notation on the time axis: shared 1e-3 exponent at the corner
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep() -> None:
    original = {p: p.read_text() for p in (JOU, YAML_CLAMPED, YAML_FREE)}
    try:
        for robin in ROBIN_PARAMS:
            robin_dir = ROOT / f"robin_{robin:.1e}"
            robin_dir.mkdir(exist_ok=True)

            rewrite_robin(YAML_CLAMPED, robin)
            rewrite_robin(YAML_FREE,    robin)

            for frac in OVERLAP_FRACTIONS:
                # symmetric overlap: both subdomains span (L + overlap) / 2
                domain_length = CANTILEVER_LENGTH * (1.0 + frac) / 2.0
                curves: list[tuple[str, np.ndarray]] = []
                out = robin_dir / f"mesh_ratios-{frac}-overlap.png"

                for ratio in MESH_RATIOS:
                    h_free = H_CLAMPED * ratio

                    print(f"\n=== robin={robin:.1e}  overlap={frac*100:.0f}%  "
                          f"h_free/h_clamped={ratio:g} ===", flush=True)

                    rewrite_jou(h_clamped=H_CLAMPED,
                                h_free=h_free,
                                domain_length_clamped=domain_length,
                                domain_length_free=domain_length,
                                length=CANTILEVER_LENGTH)

                    cleanup_stage_outputs()

                    run(["julia", "journal-to-exodus.jl", JOU.name])
                    run(["/home/lukasostien/Repos/Norma.jl/bin/diff", YAML_MULTI.name])

                    x = load_total_energy(ROOT / "cantilever-multi-energy.csv")
                    if len(x) == 0:
                        raise RuntimeError("no rows in cantilever-multi-energy.csv -- did norma write outputs?")
                    if x[0] == 0.0:
                        raise RuntimeError("E(0) == 0; drift normalization would divide by zero")
                    drift = (x / x[0]) * 100.0 - 100.0
                    curves.append((f"{ratio:.2f}", drift))

                    cleanup_stage_outputs()

                    # regenerate the (robin, overlap) plot after every mesh-ratio
                    # so partial results are on disk if the sweep is interrupted.
                    save_overlap_plot(out, curves, robin, frac)
                    print(f"wrote {out.relative_to(ROOT)} ({len(curves)}/{len(MESH_RATIOS)} ratios)",
                          flush=True)

    finally:
        for path, text in original.items():
            path.write_text(text)
        cleanup_stage_outputs()


if __name__ == "__main__":
    try:
        sweep()
    except subprocess.CalledProcessError as e:
        print(f"\ncommand failed with exit code {e.returncode}: {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
