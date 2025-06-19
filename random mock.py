#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ---------- 0. 配置参数 ----------
N_POINTS  = 3000
BOX_SIZE  = 10.0   # deg
INNER_SIZE = 5.0   # deg
MAG_OUT, MAG_IN = 20.0, 19.8
SEED = 42

# angular separation bins (deg)
THETA_MIN, THETA_MAX = 0.0, 5.0  # deg
N_BINS = 25
bin_edges   = np.linspace(THETA_MIN, THETA_MAX, N_BINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ---------- 1. 生成随机 catalogue ----------
rng  = np.random.default_rng(SEED)
ra   = rng.uniform(0, BOX_SIZE, N_POINTS)
dec  = rng.uniform(0, BOX_SIZE, N_POINTS)
mag  = np.full(N_POINTS, MAG_OUT)

# ---------- 2. 修改中心亮区星等 ----------
center_ra, center_dec = BOX_SIZE/2, BOX_SIZE/2
radius = INNER_SIZE/2                     # 2.5°
mask_in = np.hypot(ra - center_ra, dec - center_dec) <= radius
mag[mask_in] = MAG_IN

# ---------- 3. 计算 mark (暗 ⇒ 权重大) ----------
# 外圈比中心暗 0.2mag; mark ≈ 10^(0.4×0.2) ≈1.20 
marks = 10.0 ** (0.4 * (mag - MAG_IN)) # 亮区 mark=1，暗区 mark>1
mean_m = marks.mean()  # ⟨m⟩ 用于归一化

# ---------- 4. 预分配配对统计量 ----------
pair_counts       = np.zeros(N_BINS, dtype=np.int64)
sum_w             = np.zeros(N_BINS, dtype=np.float64)  # Σ w
sum_w2            = np.zeros(N_BINS, dtype=np.float64)  # Σ w²

# ---------- 5. 预计算三角函数 ----------
ra_r  = np.deg2rad(ra)
dec_r = np.deg2rad(dec)
sin_d, cos_d = np.sin(dec_r), np.cos(dec_r)

# ---------- 6. 双循环(外层)+向量化(内层) ----------
for i in range(N_POINTS-1):
    dra      = ra_r[i+1:] - ra_r[i]    # Δα
    cos_dra  = np.cos(dra)
    cos_th   = sin_d[i]*sin_d[i+1:] + cos_d[i]*cos_d[i+1:]*cos_dra  # 球面余弦公式
    cos_th   = np.clip(cos_th, -1., 1.)  # 数值安全
    th_deg   = np.rad2deg(np.arccos(cos_th))   # θ (deg)

    w        = marks[i] * marks[i+1:]  # m_i m_j
    idx      = np.floor((th_deg-THETA_MIN)/(THETA_MAX-THETA_MIN)*N_BINS).astype(int)
    sel      = (idx>=0)&(idx<N_BINS)  # 仅保留合法 θ
    if np.any(sel):
        iv = idx[sel]  # 有效 bin
        np.add.at(pair_counts, iv, 1)
        np.add.at(sum_w,      iv, w[sel])
        np.add.at(sum_w2,     iv, w[sel]**2)

# ---------- 7. 计算 M(θ) 及误差 ----------
M_theta = np.full(N_BINS, np.nan)
err_M   = np.full(N_BINS, np.nan)

valid = pair_counts > 0
M_theta[valid] = (sum_w[valid]/pair_counts[valid]) / (mean_m**2)

# 误差：需至少 2 个配对
valid_var = pair_counts > 1
if np.any(valid_var):
    mu   = sum_w[valid_var]/pair_counts[valid_var]
    var  = (sum_w2[valid_var]/pair_counts[valid_var] - mu**2) \
           * pair_counts[valid_var]/(pair_counts[valid_var]-1)               # 无偏估计
    err_mu = np.sqrt(var/pair_counts[valid_var])
    err_M[valid_var] = err_mu / (mean_m**2)

# ---------- 8. 输出并绘图 ----------
print("# theta_centre(deg)   M(theta)   err")
for t, mval, e in zip(bin_centres, M_theta, err_M):
    print(f"{t:13.5f}  {mval:10.5f}  {e:10.5f}")

plt.figure(figsize=(6,4))
plt.errorbar(bin_centres[valid], M_theta[valid],
             yerr=err_M[valid], fmt='o', capsize=3, label='Angular M(θ)')
plt.axhline(1.0, ls='--', c='k', alpha=0.4)
plt.xlabel(r"$\theta$  (deg)")
plt.ylabel(r"$M(\theta)$")
plt.title("Angular marked correlation function with errors")
plt.legend()
plt.tight_layout()
plt.show()
