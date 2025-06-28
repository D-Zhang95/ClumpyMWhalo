#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""marked_correlation_module.py – Compare constant‑patch vs NFW‑patch catalogues
================================================================================
This module builds two synthetic star/galaxy catalogues and plots their *angular
marked correlation functions* on the same figure.

Catalogues
----------
1. **ConstantPatchCatalog** – an inner disc that is *uniformly* brighter by a
   fixed Δm.
2. **NFWPatchCatalog** – an inner disc whose brightness follows an NFW‑inspired
   radial profile, matching the outer magnitude at the disc edge.

Quick usage
-----------
```bash
# compute and visualise the two M(θ) curves
python marked_correlation_module.py

# run the built‑in unit tests
python marked_correlation_module.py --test
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Helper type alias
# ----------------------------------------------------------------------------
MagFunc = Callable[[np.ndarray], np.ndarray]  # r (deg) → magnitude

# ----------------------------------------------------------------------------
# Base catalogue class
# ----------------------------------------------------------------------------

@dataclass
class BaseCatalog:
    """Synthetic catalogue base class; subclasses override *_compute_mag*."""

    n_points: int = 3000
    box_size: float = 10.0            # deg (square FoV)
    inner_size: float = 5.0           # deg (diameter of bright patch)
    mag_out: float = 20.0             # outer‑region magnitude
    delta_mag0: float = 0.2           # central brightening (mag)
    seed: int | None = 42

    # populated at runtime ---------------------------------------------------
    ra: np.ndarray = field(init=False, repr=False)
    dec: np.ndarray = field(init=False, repr=False)
    mag: np.ndarray = field(init=False, repr=False)
    marks: np.ndarray = field(init=False, repr=False)
    mean_mark: float = field(init=False)

    # ---------------------------------------------------------------------
    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.ra = rng.uniform(0.0, self.box_size, self.n_points)
        self.dec = rng.uniform(0.0, self.box_size, self.n_points)

        # radial distance from field centre
        centre = np.array([self.box_size / 2, self.box_size / 2])
        r = np.hypot(self.ra - centre[0], self.dec - centre[1])
        radius = self.inner_size / 2

        # initialise magnitudes to outer value
        self.mag = np.full(self.n_points, self.mag_out, dtype=float)
        mask_in = r <= radius
        if np.any(mask_in):
            self.mag[mask_in] = self._compute_mag(r[mask_in])

        # marks relative to brightest expected magnitude (mag_ref)
        mag_ref = self.mag_out - self.delta_mag0  # 19.8 in default config
        self.marks = 10.0 ** (0.4 * (self.mag - mag_ref))
        self.mean_mark = float(self.marks.mean())

    # ------------------------------------------------------------------
    def _compute_mag(self, r: np.ndarray) -> np.ndarray:  # to be overridden
        raise NotImplementedError

    # ------------------------------------------------------------------
    def to_radians(self):
        return np.deg2rad(self.ra), np.deg2rad(self.dec)


# ----------------------------------------------------------------------------
# 1) Constant bright patch catalogue
# ----------------------------------------------------------------------------

class ConstantPatchCatalog(BaseCatalog):
    """Inner disc uniformly brighter by *delta_mag0*."""

    def _compute_mag(self, r: np.ndarray) -> np.ndarray:  # noqa: D401
        return np.full_like(r, self.mag_out - self.delta_mag0)


# ----------------------------------------------------------------------------
# 2) NFW‑profile bright patch catalogue
# ----------------------------------------------------------------------------

class NFWPatchCatalog(BaseCatalog):
    """Inner disc brightness ∝ (1 + r/rs)⁻², normalised so Δm(radius)=0."""

    rs_ratio: float = 0.2  # rs = rs_ratio × radius

    def _compute_mag(self, r: np.ndarray) -> np.ndarray:  # noqa: D401
        radius = self.inner_size / 2
        rs = self.rs_ratio * radius
        x = r / rs
        x_edge = radius / rs

        f = (1 + x) ** -2          # projected NFW brightness profile
        f_edge = (1 + x_edge) ** -2
        f_norm = (f - f_edge) / (1.0 - f_edge)  # 0 at edge, 1 at centre
        delta_m = self.delta_mag0 * f_norm
        return self.mag_out - delta_m


# ----------------------------------------------------------------------------
# Marked correlation estimator
# ----------------------------------------------------------------------------

class MarkedCorrelation:
    """Compute angular marked correlation function M(θ)."""

    def __init__(self, catalog: BaseCatalog, theta_min: float = 0.0, theta_max: float = 5.0, n_bins: int = 25):
        self.catalog = catalog
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_bins = n_bins

        self.bin_edges = np.linspace(theta_min, theta_max, n_bins + 1)
        self.bin_centres = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        self._compute()

    # ------------------------------------------------------------------
    def _compute(self):
        pc = np.zeros(self.n_bins, dtype=np.int64)  # pair counts
        sw = np.zeros(self.n_bins, dtype=float)     # Σw
        sw2 = np.zeros(self.n_bins, dtype=float)    # Σw²

        ra_r, dec_r = self.catalog.to_radians()
        sin_d, cos_d = np.sin(dec_r), np.cos(dec_r)
        marks = self.catalog.marks

        for i in range(self.catalog.n_points - 1):
            dra = ra_r[i + 1 :] - ra_r[i]
            cos_dra = np.cos(dra)
            cos_th = sin_d[i] * sin_d[i + 1 :] + cos_d[i] * cos_d[i + 1 :] * cos_dra
            cos_th = np.clip(cos_th, -1.0, 1.0)
            th_deg = np.rad2deg(np.arccos(cos_th))

            w = marks[i] * marks[i + 1 :]
            idx = np.floor((th_deg - self.theta_min) / (self.theta_max - self.theta_min) * self.n_bins).astype(int)
            sel = (idx >= 0) & (idx < self.n_bins)
            if np.any(sel):
                iv = idx[sel]
                np.add.at(pc, iv, 1)
                np.add.at(sw, iv, w[sel])
                np.add.at(sw2, iv, w[sel] ** 2)

        self.pair_counts = pc
        self._finalise(sw, sw2)

    # ------------------------------------------------------------------
    def _finalise(self, sw: np.ndarray, sw2: np.ndarray):
        mean_m = self.catalog.mean_mark
        self.M_theta = np.full(self.n_bins, np.nan)
        self.err_M = np.full(self.n_bins, np.nan)

        valid = self.pair_counts > 0
        self.M_theta[valid] = (sw[valid] / self.pair_counts[valid]) / (mean_m**2)

        valid_var = self.pair_counts > 1
        if np.any(valid_var):
            mu = sw[valid_var] / self.pair_counts[valid_var]
            var = (sw2[valid_var] / self.pair_counts[valid_var] - mu**2) * self.pair_counts[valid_var] / (
                self.pair_counts[valid_var] - 1
            )
            err_mu = np.sqrt(var / self.pair_counts[valid_var])
            self.err_M[valid_var] = err_mu / (mean_m**2)

    # ------------------------------------------------------------------
    def plot(self, ax: plt.Axes | None = None, label: str | None = None, **eb_kw):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        valid = self.pair_counts > 0
        default_kw = dict(fmt="o", capsize=3)
        default_kw.update(eb_kw)
        ax.errorbar(self.bin_centres[valid], self.M_theta[valid], yerr=self.err_M[valid], label=label, **default_kw)
        return ax


# ----------------------------------------------------------------------------
# Lightweight unit tests
# ----------------------------------------------------------------------------

def _run_tests():
    import unittest

    class TestCatalogues(unittest.TestCase):
        def test_mag_bounds(self):
            for Cat in (ConstantPatchCatalog, NFWPatchCatalog):
                cat = Cat(n_points=500, seed=0)
                self.assertTrue(np.all(cat.mag <= cat.mag_out))
                self.assertAlmostEqual(np.max(cat.mag), cat.mag_out)

        def test_marked_correlation_runs(self):
            for Cat in (ConstantPatchCatalog, NFWPatchCatalog):
                cat = Cat(n_points=300, seed=1)
                mc = MarkedCorrelation(cat, n_bins=12)
                self.assertTrue(np.isfinite(mc.M_theta[mc.pair_counts > 0]).all())

        def test_latex_labels(self):
            # validate LaTeX labels render without ValueError
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\theta$ (deg)")
            ax.set_ylabel(r"$M(\theta)$")
            fig.canvas.draw()  # should not raise
            plt.close(fig)

    unittest.main(argv=["ignored", "-v"], exit=False)


# ----------------------------------------------------------------------------
# CLI / plotting demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare constant vs NFW patch catalogues")
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit")
    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        # identical parameters for fair comparison
        const_cat = ConstantPatchCatalog()
        nfw_cat = NFWPatchCatalog()

        mc_const = MarkedCorrelation(const_cat)
        mc_nfw = MarkedCorrelation(nfw_cat)

        fig, ax = plt.subplots(figsize=(6, 4))
        mc_const.plot(ax=ax, label="Constant Δm = 0.2 mag", fmt="s")
        mc_nfw.plot(ax=ax, label="NFW profile", fmt="o")
        ax.axhline(1.0, ls="--", c="k", alpha=0.4)
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel(r"$M(\theta)$")
        ax.set_title("Angular marked correlation comparison")
        ax.legend()
        plt.tight_layout()
        plt.show()
