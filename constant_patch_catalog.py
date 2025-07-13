from __future__ import annotations
import numpy as np
from catalog_base import BaseCatalog

class ConstantPatchCatalog(BaseCatalog):
    """圆盘内恒定增亮 Δm₀。"""

    def _compute_mag(self, r: np.ndarray) -> np.ndarray:      # noqa: D401
        return np.full_like(r, self.mag_out - self.delta_mag0)
