#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_marked_correlation.py — user‑friendly CLI for marked‑correlation demo
=============================================================================
Compute *angular marked‑correlation functions* for synthetic catalogues and
**always save** the resulting plot as a PDF.  Users can clearly choose which
catalogue(s) to run and tweak every relevant parameter via command‑line flags.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from constant_patch_catalog import ConstantPatchCatalog
from nfw_patch_catalog import NFWPatchCatalog
from marked_correlation import MarkedCorrelation

# -----------------------------------------------------------------------------
# Catalogue factory
# -----------------------------------------------------------------------------

def build_catalog(name: str, args: argparse.Namespace):
    """Instantiate the requested catalogue with parameters from *args*."""
    common_kw = dict(
        n_points=args.n_points,
        box_size=args.box_size,
        inner_size=args.inner_size,
        mag_out=args.mag_out,
        delta_mag0=args.delta_mag0,
        seed=args.seed,
    )
    if name == "constant":
        return ConstantPatchCatalog(**common_kw)
    if name == "nfw":
        return NFWPatchCatalog(rs_ratio=args.rs_ratio, **common_kw)
    raise ValueError(f"Unknown catalogue type: {name!r}")


# -----------------------------------------------------------------------------
# CLI builder
# -----------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    epilog = textwrap.dedent(
        """Examples::\n\n"
        "  python run_marked_correlation.py                          # both catalogues\n"
        "  python run_marked_correlation.py --catalog nfw            # NFW only\n"
        "  python run_marked_correlation.py --catalog constant \\   # constant only\n"
        "         --n-points 8000 --inner-size 8\n"""
    )

    parser = argparse.ArgumentParser(
        description="Compute/save marked‑correlation functions (PDF)",
        formatter_class=lambda *a, **k: argparse.RawDescriptionHelpFormatter(*a, **k),
        epilog=epilog,
    )

    cat = parser.add_argument_group("Catalogue choice")
    cat.add_argument("--catalog", choices=["constant", "nfw", "both"], default="both")

    common = parser.add_argument_group("Common catalogue parameters")
    common.add_argument("--n-points", type=int, default=3000)
    common.add_argument("--box-size", type=float, default=10.0)
    common.add_argument("--inner-size", type=float, default=5.0)
    common.add_argument("--mag-out", type=float, default=20.0)
    common.add_argument("--delta-mag0", type=float, default=0.2)
    common.add_argument("--seed", type=int, default=42)

    nfw = parser.add_argument_group("NFW parameters")
    nfw.add_argument("--rs-ratio", type=float, default=0.2)

    mc = parser.add_argument_group("Marked‑correlation parameters")
    mc.add_argument("--theta-min", type=float, default=0.0)
    mc.add_argument("--theta-max", type=float, default=5.0)
    mc.add_argument("--n-bins", type=int, default=25)

    out = parser.add_argument_group("Output")
    out.add_argument("--out", type=str, default="marked_correlation.pdf")
    out.add_argument("--no-legend", action="store_true")

    return parser


# -----------------------------------------------------------------------------
# Summary printout helper
# -----------------------------------------------------------------------------

def print_summary(args: argparse.Namespace):
    border = "=" * 60
    print(border)
    print("RUN CONFIGURATION")
    print(border)
    for k, v in vars(args).items():
        print(f" {k:<12}: {v}")
    print(border)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()
    print_summary(args)

    names = [args.catalog] if args.catalog in {"constant", "nfw"} else ["constant", "nfw"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        cat = build_catalog(name, args)
        mc = MarkedCorrelation(cat, theta_min=args.theta_min, theta_max=args.theta_max, n_bins=args.n_bins)
        fmt = "s" if name == "constant" else "o"
        label = None if args.no_legend else ("Constant Δm" if name == "constant" else "NFW profile")
        mc.plot(ax=ax, label=label, fmt=fmt)

    ax.axhline(1.0, ls="--", c="k", alpha=0.4)
    ax.set_xlabel(r"$\theta$ (deg)")        
    ax.set_ylabel(r"$M(\theta)$")           
    if not args.no_legend:
        ax.legend()
    ax.set_title("Angular marked correlation comparison")
    plt.tight_layout()

    out_path = Path(args.out).expanduser().with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
