#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_marked_correlation.py — compare three ConstantPatchCatalog variants
=============================================================================
This version *only* uses ConstantPatchCatalog but with three inner‑disc sizes
and central magnitudes:

1. inner diameter 5°  → centre 19.8 mag (Δm = 0.2)
2. inner diameter 2°  → centre 19.9 mag (Δm = 0.1)
3. inner diameter 1°  → centre 19.95 mag (Δm = 0.05)

All three marked‑correlation curves are plotted on the same figure and saved as
a PDF (default `marked_correlation_constant_variants.pdf`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from constant_patch_catalog import ConstantPatchCatalog
from marked_correlation import MarkedCorrelation

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare three ConstantPatchCatalog variants and save PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-points", type=int, default=3000, help="Objects per catalogue")
    p.add_argument("--box-size", type=float, default=10.0, help="Field side length (deg)")
    p.add_argument("--mag-out", type=float, default=20.0, help="Outer-region magnitude")
    p.add_argument("--theta-min", type=float, default=0.0)
    p.add_argument("--theta-max", type=float, default=5.0)
    p.add_argument("--n-bins",  type=int,   default=25)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out",      type=str,   default="marked_correlation_constant_variants.pdf",
                   help="Output PDF filename")
    return p


# -----------------------------------------------------------------------------
# Build three catalogues (hard‑coded spec)
# -----------------------------------------------------------------------------

def build_catalogues(args):
    specs = [
        dict(inner_size=5.0, delta_mag0=0.2, label="D=5°; mag 19.8", fmt="o"),
        dict(inner_size=2.0, delta_mag0=0.1, label="D=2°; mag 19.9", fmt="s"),
        dict(inner_size=1.0, delta_mag0=0.05, label="D=1°; mag 19.95", fmt="^"),
    ]
    cats = []
    for spec in specs:
        cat = ConstantPatchCatalog(
            n_points=args.n_points,
            box_size=args.box_size,
            inner_size=spec["inner_size"],
            mag_out=args.mag_out,
            delta_mag0=spec["delta_mag0"],
            seed=args.seed,
        )
        cats.append((cat, spec["label"], spec["fmt"]))
    return cats


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()

    cats = build_catalogues(args)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cat, label, fmt in cats:
        mc = MarkedCorrelation(cat, theta_min=args.theta_min, theta_max=args.theta_max, n_bins=args.n_bins)
        mc.plot(ax=ax, label=label, fmt=fmt)

    ax.axhline(1.0, ls="--", c="k", alpha=0.4)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$M(\theta)$")
    ax.set_title("Marked correlation – Constant patch variants")
    ax.legend()
    plt.tight_layout()

    out_path = Path(args.out).expanduser().with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
