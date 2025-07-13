from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

MagFunc = Callable[[np.ndarray], np.ndarray]         # r (deg) → magnitude

@dataclass
class BaseCatalog:
    """Synthetic catalogue base class; subclasses override *_compute_mag*."""

    # --- public配置参数 ---
    n_points: int = 3000
    box_size: float = 10.0           # 方形视场边长 (deg)
    inner_size: float = 5.0          # 亮斑直径 (deg)
    mag_out: float = 20.0            # 外环星等
    delta_mag0: float = 0.2          # 圆心最大变亮量 (mag)
    seed: Optional[int] = 42         # RNG 种子

    # --- 运行时生成 ---
    ra:   np.ndarray = field(init=False, repr=False)
    dec:  np.ndarray = field(init=False, repr=False)
    mag:  np.ndarray = field(init=False, repr=False)
    marks: np.ndarray = field(init=False, repr=False)
    mean_mark: float = field(init=False)

    # ------------------------------------------------------------------
    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.ra  = rng.uniform(0.0, self.box_size, self.n_points)
        self.dec = rng.uniform(0.0, self.box_size, self.n_points)

        centre = np.array([self.box_size / 2, self.box_size / 2])
        r      = np.hypot(self.ra - centre[0], self.dec - centre[1])
        radius = self.inner_size / 2

        # 外环基准星等
        self.mag = np.full(self.n_points, self.mag_out, dtype=float)

        mask_in = r <= radius
        if np.any(mask_in):
            self.mag[mask_in] = self._compute_mag(r[mask_in])

        # mark = 10^{0.4 (mag – mag_ref)}
        mag_ref      = self.mag_out - self.delta_mag0
        self.marks   = 10.0 ** (0.4 * (self.mag - mag_ref))
        self.mean_mark = float(self.marks.mean())

    # 子类重写：给定半径数组，返回对应亮斑内星等 -------------------------
    def _compute_mag(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # 便于后续角距计算
    def to_radians(self):
        return np.deg2rad(self.ra), np.deg2rad(self.dec)
