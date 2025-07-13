from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from catalog_base import BaseCatalog

class MarkedCorrelation:
    """Angular marked correlation function M(θ)."""

    # ----------------------------------------------------------
    def __init__(self,
                 catalog:  BaseCatalog,
                 theta_min: float = 0.0,
                 theta_max: float = 5.0,
                 n_bins:    int   = 25):
        self.catalog   = catalog
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_bins    = n_bins
        self.bin_edges   = np.linspace(theta_min, theta_max, n_bins + 1)
        self.bin_centres = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self._compute()

    # ----------------------------------------------------------
    def _compute(self):
        pc  = np.zeros(self.n_bins, dtype=np.int64)  # pair counts
        sw  = np.zeros(self.n_bins)                  # Σw
        sw2 = np.zeros(self.n_bins)                  # Σw²

        ra_r, dec_r  = self.catalog.to_radians()
        sin_d, cos_d = np.sin(dec_r), np.cos(dec_r)
        marks        = self.catalog.marks

        for i in range(self.catalog.n_points - 1):
            dra      = ra_r[i + 1:] - ra_r[i]
            cos_dra  = np.cos(dra)
            cos_th   = sin_d[i] * sin_d[i + 1:] + cos_d[i] * cos_d[i + 1:] * cos_dra
            cos_th   = np.clip(cos_th, -1.0, 1.0)
            th_deg   = np.rad2deg(np.arccos(cos_th))

            w   = marks[i] * marks[i + 1:]
            idx = np.floor((th_deg - self.theta_min) /
                           (self.theta_max - self.theta_min) * self.n_bins).astype(int)
            sel = (idx >= 0) & (idx < self.n_bins)
            if np.any(sel):
                iv = idx[sel]
                np.add.at(pc,  iv, 1)
                np.add.at(sw,  iv, w[sel])
                np.add.at(sw2, iv, w[sel] ** 2)

        self.pair_counts = pc
        self._finalise(sw, sw2)

    # ----------------------------------------------------------
    def _finalise(self, sw: np.ndarray, sw2: np.ndarray):
        mean_m      = self.catalog.mean_mark
        self.M_theta = np.full(self.n_bins, np.nan)
        self.err_M   = np.full(self.n_bins, np.nan)

        valid = self.pair_counts > 0
        self.M_theta[valid] = (sw[valid] / self.pair_counts[valid]) / (mean_m ** 2)

        valid_var = self.pair_counts > 1
        if np.any(valid_var):
            mu     = sw[valid_var] / self.pair_counts[valid_var]
            var    = (sw2[valid_var] / self.pair_counts[valid_var] - mu**2) \
                     * self.pair_counts[valid_var] / (self.pair_counts[valid_var] - 1)
            err_mu = np.sqrt(var / self.pair_counts[valid_var])
            self.err_M[valid_var] = err_mu / (mean_m ** 2)

    # ----------------------------------------------------------
    def plot(self, ax: plt.Axes | None = None, label: str | None = None, **eb_kw):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        valid = self.pair_counts > 0
        default_kw = dict(fmt='o', capsize=3)
        default_kw.update(eb_kw)
        ax.errorbar(self.bin_centres[valid], self.M_theta[valid],
                    yerr=self.err_M[valid], label=label, **default_kw)
        return ax
