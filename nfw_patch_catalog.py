from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from catalog_base import BaseCatalog

@dataclass
class NFWPatchCatalog(BaseCatalog):
    """亮度 ∝ (1 + r/rs)^{-2}，并在圆盘边缘与外环星等连续。"""

    rs_ratio: float = 0.2            # rs = rs_ratio × (radius)

    def _compute_mag(self, r: np.ndarray) -> np.ndarray:      # noqa: D401
        radius = self.inner_size / 2
        rs     = self.rs_ratio * radius
        x      = r / rs
        x_edge = radius / rs

        f      = (1 + x) ** -2
        f_edge = (1 + x_edge) ** -2
        f_norm = (f - f_edge) / (1.0 - f_edge)               # 0@edge, 1@centre
        delta_m = self.delta_mag0 * f_norm
        return self.mag_out - delta_m
