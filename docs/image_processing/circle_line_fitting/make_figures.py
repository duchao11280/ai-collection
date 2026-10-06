#!/usr/bin/env python3
"""Sinh toàn bộ ảnh debug cho bài circle_line_fitting.md.

Chạy:  python make_figures.py      (cần numpy, matplotlib, opencv-python)
Ảnh được ghi vào ./images/, các bảng số được in ra màn hình.
"""
from __future__ import annotations

import math
import os
import time

import cv2
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle as CirclePatch, Rectangle  # noqa: E402

import numpy as np                                   # noqa: E402

import fitting as F                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Kiểu dáng chung: màu theo "thực thể" (mỗi phương pháp giữ một màu ở mọi hình)
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C = {
    "truth": INK, "ols": "#eb6834", "kasa": "#eb6834", "tls": "#2a78d6",
    "taubin": "#2a78d6", "geometric": "#1baf7a", "trimmed": "#4a3aa7",
    "ransac": "#008300", "tukey": "#eda100", "tukey_kasa": "#e87ba4",
    "outlier": "#e34948",
}
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
WEIGHT_CMAP = LinearSegmentedColormap.from_list("w", ["#e1e0d9", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "text.color": INK, "font.size": 9.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 110,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.relpath(path, HERE))


def draw_line(ax, fit_or_p0, direction=None, span=(-400, 400), **kw):
    if direction is None:
        p0, d = fit_or_p0.p0, fit_or_p0.direction
    else:
        p0, d = np.asarray(fit_or_p0), np.asarray(direction)
    t = np.array(span, dtype=float)
    ax.plot(p0[0] + t * d[0], p0[1] + t * d[1], **kw)


def draw_circle(ax, cx, cy, r, **kw):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), **kw)


def pts_in(ax, p, color, label=None, s=14, zorder=3):
    ax.scatter(p[:, 0], p[:, 1], s=s, color=color, edgecolors=SURFACE, linewidths=0.6,
               label=label, zorder=zorder)


def pts_out(ax, p, label=None, color=MUTED, s=22, zorder=3):
    ax.scatter(p[:, 0], p[:, 1], s=s, marker="x", color=color, linewidths=1.2,
               label=label, zorder=zorder)


def equal(ax, xlim=None, ylim=None):
    ax.set_aspect("equal", adjustable="box")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


# ===========================================================================
# Dữ liệu tổng hợp
# ===========================================================================
def line_points(rng, n, p_start, p_end, sigma):
    t = rng.uniform(0, 1, n)
    p = np.asarray(p_start) + t[:, None] * (np.asarray(p_end) - np.asarray(p_start))
    return p + rng.normal(0, sigma, (n, 2))


def arc_points(rng, n, cx, cy, r, a0_deg, a1_deg, sigma, even=False):
    if even:
        t = np.deg2rad(np.linspace(a0_deg, a1_deg, n))
    else:
        t = np.deg2rad(rng.uniform(a0_deg, a1_deg, n))
    p = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
    return p + rng.normal(0, sigma, (n, 2))


# ===========================================================================
# Ảnh 02: bình phương tối thiểu thông thường so với Total Least Squares
# ===========================================================================
def fig_line_ols_tls():
    print("[02] OLS vs TLS")
    rng = np.random.default_rng(3)
    p = line_points(rng, 18, (10, 20), (90, 65), 3.0)
    a, b = F.fit_line_ols(p)
    tls = F.fit_line_tls(p)

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axs[0]
    xs = np.array([0, 100])
    ax.plot(xs, a * xs + b, color=C["ols"], label="y = a·x + b")
    for x, y in p:
        ax.plot([x, x], [y, a * x + b], color=C["ols"], lw=1, alpha=0.8)
    pts_in(ax, p, INK2, label="điểm biên")
    ax.set_title("(a) Sai số theo phương thẳng đứng")
    equal(ax, (0, 100), (0, 100))
    ax.legend(loc="upper left")

    ax = axs[1]
    draw_line(ax, tls, color=C["tls"], label="Total Least Squares")
    n = tls.normal
    for q in p:
        d = (q - tls.p0) @ n
        foot = q - d * n
        ax.plot([q[0], foot[0]], [q[1], foot[1]], color=C["tls"], lw=1, alpha=0.8)
    pts_in(ax, p, INK2)
    ax.set_title("(b) Sai số trực giao (khoảng cách thật)")
    equal(ax, (0, 100), (0, 100))
    ax.legend(loc="upper left")

    ax = axs[2]
    rng = np.random.default_rng(11)
    q = line_points(rng, 30, (50, 5), (51, 95), 2.0)          # cạnh gần thẳng đứng
    a2, b2 = F.fit_line_ols(q)
    tls2 = F.fit_line_tls(q)
    ys = np.array([0, 100])
    ax.plot((ys - b2) / a2 if abs(a2) > 1e-9 else [0, 0], ys, color=C["ols"],
            ls="--", label=f"y = a·x + b  (a = {a2:.2f})")
    draw_line(ax, tls2, color=C["tls"], label="Total Least Squares")
    pts_in(ax, q, INK2)
    ang = math.degrees(math.atan2(tls2.direction[1], tls2.direction[0])) % 180
    ang_ols = math.degrees(math.atan(a2)) % 180
    ax.set_title(f"(c) Cạnh gần đứng (thật 89.4°)\ny = a·x + b: {ang_ols:.1f}°,  Total Least Squares: {ang:.1f}°")
    equal(ax, (0, 100), (0, 100))
    ax.legend(loc="lower right")
    save(fig, "02_line_ols_vs_tls.png")


# ===========================================================================
# Ảnh 03: hình học của Total Least Squares (phân tích giá trị suy biến)
# ===========================================================================
def fig_line_tls_svd():
    print("[03] TLS / SVD geometry")
    rng = np.random.default_rng(4)
    p = line_points(rng, 60, (15, 25), (85, 60), 2.5)
    fit = F.fit_line_tls(p)
    s = fit.info["singular_values"]
    sd = s / math.sqrt(len(p))                       # độ lệch chuẩn theo mỗi trục chính
    d = fit.direction if fit.direction[0] >= 0 else -fit.direction   # cho mũi tên hướng sang phải
    n = np.array([-d[1], d[0]])

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    pts_in(ax, p, INK2, s=12)
    draw_line(ax, fit, color=C["tls"], lw=1.5, alpha=0.9)
    t = np.linspace(0, 2 * np.pi, 200)
    ell = fit.p0 + 2 * (np.outer(np.cos(t), d * sd[0]) + np.outer(np.sin(t), n * sd[1]))
    ax.plot(ell[:, 0], ell[:, 1], color=MUTED, lw=1)
    kw = dict(head_width=1.6, length_includes_head=True, lw=1.8, zorder=5)
    ax.arrow(*fit.p0, *(d * 2 * sd[0]), color=C["tls"], **kw)
    ax.arrow(*fit.p0, *(n * 2 * sd[1] * 3), color=INK2, **kw)
    ax.scatter(*fit.p0, s=60, color=INK, zorder=6, edgecolors=SURFACE, linewidths=1.5)
    ax.annotate("trọng tâm p0", fit.p0, xytext=(10, -14), textcoords="offset points", fontsize=9)
    arr = dict(arrowstyle="-", color=MUTED, lw=0.8)
    ax.annotate(f"vt[0]: chỉ phương, σ1 = {sd[0]:.1f}", fit.p0 + d * 2 * sd[0], xytext=(60, 20),
                textcoords="data", fontsize=9, arrowprops=arr)
    ax.annotate(f"vt[1]: pháp tuyến (vẽ ×3)\nσ2 = {sd[1]:.2f} = căn quân phương phần dư",
                fit.p0 + n * 6 * sd[1], xytext=(8, 64), textcoords="data", fontsize=9, arrowprops=arr)
    ax.set_title("(a) Hai vector suy biến phải của dữ liệu đã trừ trọng tâm")
    equal(ax, (5, 95), (10, 75))

    ax = axs[1]
    th = np.linspace(0, np.pi, 361)
    q = p - fit.p0
    J = [np.mean((q @ np.array([math.cos(a), math.sin(a)])) ** 2) for a in th]
    ax.plot(np.degrees(th), J, color=C["tls"])
    th_n = math.degrees(math.atan2(n[1], n[0])) % 180
    th_d = math.degrees(math.atan2(d[1], d[0])) % 180
    ax.axvline(th_n, color=MUTED, lw=1)
    ax.axvline(th_d, color=MUTED, lw=1)
    ax.annotate(f"cực tiểu = σ2² = {sd[1] ** 2:.2f}\n(pháp tuyến, {th_n:.0f}°)",
                (th_n, sd[1] ** 2), xytext=(10, 25), textcoords="offset points", fontsize=9)
    ax.annotate(f"cực đại = σ1² = {sd[0] ** 2:.0f}\n(chỉ phương, {th_d:.0f}°)",
                (th_d, sd[0] ** 2), xytext=(10, -30), textcoords="offset points", fontsize=9)
    ax.set_xlabel("góc của pháp tuyến n (độ)")
    ax.set_ylabel("mean( ((p − p0)·n)² )")
    ax.set_title("(b) Hàm mục tiêu theo hướng pháp tuyến")
    save(fig, "03_line_tls_svd.png")


# ===========================================================================
# Ảnh 04: RANSAC cho đường thẳng
# ===========================================================================
def line_dataset(seed=21):
    rng = np.random.default_rng(seed)
    inl = line_points(rng, 60, (10, 30), (190, 120), 0.8)
    seg = line_points(rng, 18, (40, 120), (110, 150), 0.8)           # một cạnh lân cận
    uni = rng.uniform((0, 0), (200, 170), (22, 2))
    p = np.vstack([inl, seg, uni])
    truth = np.zeros(len(p), dtype=bool)
    truth[:60] = True
    return p, truth


def fig_line_ransac():
    print("[04] RANSAC line")
    p, _ = line_dataset()
    res = F.fit_line_ransac(p, tol=2.5, iters=300)
    tls_all = F.fit_line_tls(p)
    info = res.info

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax = axs[0]
    pts_in(ax, p, MUTED, s=10)
    for j in range(0, 14):
        i0, i1 = info["pairs"][j]
        a, b = p[i0], p[i1]
        dd = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        draw_line(ax, a, dd, color=INK2, lw=0.8, alpha=0.55)
        ax.scatter(*np.vstack([a, b]).T, s=18, color=INK2, zorder=4)
    best = info["best"]
    draw_line(ax, info["best_p0"], info["best_dir"], color=C["ransac"], lw=2,
              label=f"giả thuyết tốt nhất: {info['counts'][best]} inlier")
    i0, i1 = info["pairs"][best]
    ax.scatter(*p[[i0, i1]].T, s=50, color=C["ransac"], edgecolors=SURFACE, linewidths=1.5, zorder=6)
    ax.set_title("(a) Mỗi cặp điểm = một giả thuyết")
    equal(ax, (0, 200), (0, 170))
    ax.legend(loc="upper left")

    ax = axs[1]
    n = res.normal
    t = np.array([-150, 150])
    band = np.array([res.p0 + t[0] * res.direction + 2.5 * n, res.p0 + t[1] * res.direction + 2.5 * n,
                     res.p0 + t[1] * res.direction - 2.5 * n, res.p0 + t[0] * res.direction - 2.5 * n])
    ax.fill(band[:, 0], band[:, 1], color=C["ransac"], alpha=0.12, lw=0, label="dải tol = ±2.5")
    pts_in(ax, p[res.inliers], C["ransac"], label=f"inlier ({res.inliers.sum()})")
    pts_out(ax, p[~res.inliers], label=f"outlier ({(~res.inliers).sum()})")
    draw_line(ax, tls_all, color=C["tls"], ls="--", lw=1.6, label="Total Least Squares, mọi điểm")
    draw_line(ax, res, color=C["ransac"], lw=1.6, label="RANSAC → Total Least Squares trên inlier")
    ax.set_title("(b) Fit lại trên inlier")
    equal(ax, (0, 200), (0, 170))
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[2]
    cnt = info["counts"]
    ax.scatter(np.arange(len(cnt)), cnt, s=6, color=MUTED, label="số inlier của từng giả thuyết")
    ax.plot(np.maximum.accumulate(cnt), color=C["ransac"], label="tốt nhất tới thời điểm đó")
    ax.set_xlabel("chỉ số giả thuyết")
    ax.set_ylabel("số inlier")
    ax.set_title("(c) Số inlier của từng giả thuyết")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right")
    save(fig, "04_line_ransac.png")


# ===========================================================================
# Ảnh 05: số vòng lặp RANSAC cần thiết
# ===========================================================================
def fig_ransac_iterations():
    print("[05] RANSAC iterations")
    w = np.linspace(0.2, 0.95, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for s, col, ls, name in ((2, C["tls"], "-", "mẫu 2 điểm (đường thẳng)"),
                             (3, C["kasa"], "--", "mẫu 3 điểm (đường tròn)")):
        N = [F.ransac_iterations(0.99, x, s) for x in w]
        ax.plot(w, N, color=col, ls=ls, label=name)
        for x in (0.5, 0.3):
            v = F.ransac_iterations(0.99, x, s)
            ax.scatter([x], [v], s=36, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=5)
            ax.annotate(f"{v}", (x, v), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("tỉ lệ inlier w")
    ax.set_ylabel("số vòng N (thang log)")
    ax.set_title("Số vòng lặp để có ≥ 1 mẫu sạch với xác suất 99%")
    ax.legend(loc="upper right")
    save(fig, "05_ransac_iterations.png")


# ===========================================================================
# Ảnh 06: Trimmed và Tukey cho đường thẳng
# ===========================================================================
def fig_line_trimmed_tukey():
    print("[06] trimmed / Tukey line")
    rng = np.random.default_rng(8)
    good = line_points(rng, 50, (0, 10), (100, 40), 0.7)
    burr = line_points(rng, 12, (55, 31), (75, 37), 0.5) + np.array([-1.2, 4.0]) * rng.uniform(1, 2.2, (12, 1))
    gross = rng.uniform((0, 30), (100, 60), (4, 2))
    p = np.vstack([good, burr, gross])

    trim = F.fit_line_trimmed(p, rounds=4, k=2.5, scale="mad")
    tuk = F.fit_line_tukey(p)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axs[0]
    hist = trim.info["history"]
    for i, (p0, d, keep) in enumerate(hist):
        draw_line(ax, p0, d, color=BLUES[min(i, 3)], lw=1.8 if i else 1.4,
                  ls="--" if i == 0 else "-", label=f"vòng {i}: giữ {keep.sum()}/{len(p)}")
    pts_in(ax, p[trim.inliers], C["trimmed"], label="giữ ở vòng cuối")
    pts_out(ax, p[~trim.inliers], label="bị cắt")
    ax.set_title("(a) Trimmed: fit → bỏ |d| > 2.5·s → fit lại")
    equal(ax, (0, 100), (0, 62))
    ax.legend(loc="upper left", fontsize=8)

    ax = axs[1]
    w = tuk.info["weights"]
    sc = ax.scatter(p[:, 0], p[:, 1], c=w, cmap=WEIGHT_CMAP, vmin=0, vmax=1, s=26,
                    edgecolors=INK2, linewidths=0.4, zorder=3)
    draw_line(ax, F.fit_line_tls(p), color=C["tls"], ls="--", lw=1.4, label="Total Least Squares, mọi điểm")
    draw_line(ax, tuk, color=C["tukey"], lw=1.8, label="Tukey (tái trọng số lặp)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("trọng số w", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("(b) Tukey: mỗi điểm có trọng số thay vì giữ / bỏ")
    equal(ax, (0, 100), (0, 62))
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "06_line_trimmed_tukey.png")


# ===========================================================================
# Ảnh 07: hàm mất mát và hàm trọng số
# ===========================================================================
def fig_weight_functions():
    print("[07] weight functions")
    u = np.linspace(-3, 3, 601)
    k, c = 1.345, 1.0     # trục hoành đã chia sẵn cho c*s với Tukey

    def rho_huber(x):
        a = np.abs(x)
        return np.where(a <= k, 0.5 * x * x, k * a - 0.5 * k * k)

    def rho_tukey(x):
        return np.where(np.abs(x) < 1, (1 - (1 - x * x) ** 3) / 6, 1 / 6)

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axs[0]
    ax.plot(u, 0.5 * u * u, color=C["kasa"], label="bình phương tối thiểu: u²/2")
    ax.plot(u, rho_huber(u), color=C["tls"], ls="--", label="Huber (k = 1.345)")
    ax.plot(u, rho_tukey(u) * 6, color=C["tukey"], label="Tukey (×6 để dễ nhìn)")
    ax.set_ylim(0, 3)
    ax.set_xlabel("u = phần dư / thang đo")
    ax.set_title("(a) Hàm mất mát ρ(u)")
    ax.legend(loc="upper center")

    ax = axs[1]
    ax.plot(u, np.ones_like(u), color=C["kasa"], label="bình phương tối thiểu: w = 1")
    ax.plot(u, F.huber_weights(u, k), color=C["tls"], ls="--", label="Huber: w = k/|u|")
    ax.plot(u, F.tukey_weights(u / c), color=C["tukey"], label="Tukey: w = (1 − u²)²")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("u = phần dư / (c · s)  (Tukey)   hoặc phần dư / s  (Huber)")
    ax.set_title("(b) Hàm trọng số w(u) = ρ'(u) / u")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=8)
    save(fig, "07_weight_functions.png")


# ===========================================================================
# Ảnh 08: Kasa so với Taubin trên cung ngắn
# ===========================================================================
def fig_circle_arc_bias():
    print("[08] arc bias Monte Carlo")
    R = 100.0
    rng = np.random.default_rng(12)
    p = arc_points(rng, 40, 0, 0, R, -30, 30, 1.0, even=True)
    fits = {"kasa": F.fit_circle_kasa(p), "taubin": F.fit_circle_taubin(p),
            "geometric": F.fit_circle_geometric(p)}

    spans = np.array([30, 45, 60, 90, 120, 180, 270, 360])
    methods = {
        "kasa": ("Kasa", lambda q: F.fit_circle_kasa(q)),
        "taubin": ("Taubin", lambda q: F.fit_circle_taubin(q)),
        "geometric": ("hình học (Levenberg-Marquardt)", lambda q: F.fit_circle_geometric(q)),
        "tukey_kasa": ("Taubin → Tukey, bên trong Kasa", lambda q: F.fit_circle_tukey(q, inner="kasa")),
        "tukey": ("Taubin → Tukey, bên trong Taubin", lambda q: F.fit_circle_tukey(q, inner="taubin")),
    }
    trials = 300
    stats = {k: {"bias": [], "rmse": []} for k in methods}
    for sp in spans:
        errs = {k: [] for k in methods}
        for _ in range(trials):
            q = arc_points(rng, 40, 0, 0, R, -sp / 2, sp / 2, 1.0, even=True)
            for k, (_, fn) in methods.items():
                errs[k].append(fn(q).r - R)
        for k in methods:
            e = np.array(errs[k])
            stats[k]["bias"].append(e.mean())
            stats[k]["rmse"].append(math.sqrt(np.mean(e * e)))

    print("  span  " + "  ".join(f"{k:>12s}" for k in methods))
    for i, sp in enumerate(spans):
        print(f"  {sp:4d}° " + "  ".join(
            f"{stats[k]['bias'][i]:+6.2f}/{stats[k]['rmse'][i]:5.2f}" for k in methods))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    ax = axs[0]
    draw_circle(ax, 0, 0, R, color=C["truth"], lw=1, ls=":", label=f"thật: r = {R:.0f}")
    styles = {"kasa": "-", "taubin": "-", "geometric": "--"}
    names = {"kasa": "Kasa", "taubin": "Taubin", "geometric": "hình học"}
    for k, f in fits.items():
        draw_circle(ax, f.cx, f.cy, f.r, color=C[k], ls=styles[k], lw=1.6,
                    label=f"{names[k]}: r = {f.r:.1f}")
    pts_in(ax, p, INK2, s=12, zorder=6)
    ax.set_title("(a) Cung 60°, nhiễu σ = 1 pixel")
    equal(ax, (-120, 120), (-120, 120))
    ax.legend(loc="lower left", fontsize=8)

    markers = {"kasa": "o", "taubin": "s", "geometric": "^", "tukey_kasa": "D", "tukey": "v"}
    lss = {"kasa": "-", "taubin": "-", "geometric": "--", "tukey_kasa": ":", "tukey": "-."}
    for ax, key, title, ylab in ((axs[1], "bias", "(b) Độ lệch trung bình của bán kính",
                                  "mean(r̂ − r)  (pixel)"),
                                 (axs[2], "rmse", "(c) Sai số căn quân phương của bán kính",
                                  "căn quân phương (r̂ − r)  (pixel, thang log)")):
        for k, (name, _) in methods.items():
            ax.plot(spans, stats[k][key], color=C[k], ls=lss[k], marker=markers[k], ms=5,
                    lw=1.6, label=name)
        ax.set_xlabel("độ dài cung nhìn thấy (độ)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks([30, 60, 90, 180, 270, 360])
    axs[1].axhline(0, color=AXIS, lw=1)
    axs[2].set_yscale("log")
    axs[1].legend(loc="lower right", fontsize=7.8)
    save(fig, "08_circle_arc_bias.png")
    return spans, stats


# ===========================================================================
# Ảnh 09: Trimmed cho đường tròn, từng vòng
# ===========================================================================
def circle_defect_dataset(seed=5):
    rng = np.random.default_rng(seed)
    R, cx, cy = 100.0, 0.0, 0.0
    ang = np.deg2rad(np.arange(0, 360, 4.0))
    r = np.full(len(ang), R)
    notch = (np.rad2deg(ang) >= 196) & (np.rad2deg(ang) <= 232)          # vết khuyết lõm vào
    r[notch] -= 14 * np.sin(np.linspace(0, np.pi, notch.sum())) + 4
    dust = (np.rad2deg(ang) >= 300) & (np.rad2deg(ang) <= 320)           # bụi làm biên lồi ra
    r[dust] += rng.uniform(6, 11, dust.sum())
    p = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]) + rng.normal(0, 0.8, (len(ang), 2))
    gross = np.array([[40, 30], [-20, 60], [130, 110]], dtype=float)
    p = np.vstack([p, gross])
    ang = np.concatenate([ang, np.arctan2(gross[:, 1], gross[:, 0]) % (2 * np.pi)])
    return p, ang, (cx, cy, R)


def fig_circle_trimmed():
    print("[09] trimmed circle rounds")
    p, ang, (cx0, cy0, R) = circle_defect_dataset()
    fit = F.fit_circle_trimmed(p, rounds=4, k=2.5, scale="mad")
    hist = fit.info["history"]
    n = len(hist)
    fig, axs = plt.subplots(2, n, figsize=(3.9 * n, 7.6), gridspec_kw={"height_ratios": [1.15, 1]})
    for i, (cx, cy, r, keep) in enumerate(hist):
        err = math.hypot(cx - cx0, cy - cy0)
        ax = axs[0, i]
        draw_circle(ax, cx0, cy0, R, color=C["truth"], lw=1, ls=":")
        pts_in(ax, p[keep], C["trimmed"], s=12)
        pts_out(ax, p[~keep], s=20)
        draw_circle(ax, cx, cy, r, color=C["trimmed"], lw=1.5)
        ax.scatter([cx], [cy], s=30, color=C["trimmed"], marker="+", zorder=5)
        ax.set_title(f"vòng {i}: giữ {keep.sum()}/{len(p)}\nlệch tâm {err:.2f} pixel, r = {r:.2f}")
        equal(ax, (-145, 145), (-145, 145))

        ax = axs[1, i]
        res = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r
        s = 1.4826 * np.median(np.abs(res))
        deg = np.rad2deg(ang)
        ax.axhspan(-2.5 * s, 2.5 * s, color=C["trimmed"], alpha=0.10, lw=0)
        ax.scatter(deg[keep], res[keep], s=10, color=C["trimmed"])
        ax.scatter(deg[~keep], res[~keep], s=16, marker="x", color=MUTED)
        ax.set_ylim(-25, 25)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("góc (độ)")
        if i == 0:
            ax.set_ylabel("phần dư d = ρ − r  (pixel)")
        ax.set_title(f"ngưỡng ±2.5·s = ±{2.5 * s:.2f}")
    fig.suptitle("Trimmed: vùng tô = dải giữ lại của vòng tiếp theo; vết khuyết ở 196–232°, bụi ở 300–320°",
                 x=0.01, ha="left", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "09_circle_trimmed_rounds.png")


# ===========================================================================
# Ảnh 10: đường tròn ngoại tiếp 3 điểm
# ===========================================================================
def fig_circumcircle():
    print("[10] circumcircle")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, tri, title in ((axs[0], np.array([[20.0, 30.0], [80.0, 15.0], [70.0, 70.0]]),
                            "(a) Tâm = giao hai đường trung trực"),
                           (axs[1], np.array([[10.0, 40.0], [50.0, 43.0], [90.0, 44.0]]),
                            "(b) Gần thẳng hàng: d → 0, bán kính → ∞")):
        ux, uy, r, ok = F.circumcircle(tri[0], tri[1], tri[2])
        ux, uy, r = float(ux), float(uy), float(r)
        a, b, c = tri
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        draw_circle(ax, ux, uy, r, color=C["ransac"], lw=1.6)
        for P, Q in ((a, b), (b, c)):
            mid = (P + Q) / 2
            dirv = np.array([-(Q - P)[1], (Q - P)[0]])
            dirv /= np.linalg.norm(dirv)
            draw_line(ax, mid, dirv, span=(-300, 300), color=MUTED, lw=1, ls="--")
            ax.plot(*np.vstack([P, Q]).T, color=INK2, lw=1)
            ax.scatter(*mid, s=14, color=MUTED, zorder=4)
        ax.scatter(*tri.T, s=46, color=INK, zorder=6, edgecolors=SURFACE, linewidths=1.5)
        for lab, P in zip("abc", tri):
            ax.annotate(lab, P, xytext=(6, 6), textcoords="offset points", fontsize=11)
        if abs(uy) < 1e4:
            ax.scatter([ux], [uy], s=40, color=C["ransac"], marker="+", zorder=6)
        ax.set_title(f"{title}\nd = {d:.0f},  r = {r:.1f}")
        equal(ax, (-10, 110), (-20, 100))
    save(fig, "10_circumcircle.png")


# ===========================================================================
# Ảnh 11: RANSAC cho đường tròn
# ===========================================================================
def circle_neighbor_dataset(seed=31):
    rng = np.random.default_rng(seed)
    main = arc_points(rng, 110, 0, 0, 100, -150, 150, 0.8)
    nb = arc_points(rng, 45, 150, 30, 40, 100, 330, 0.8)            # lỗ lân cận nhỏ hơn
    uni = rng.uniform((-140, -140), (200, 140), (25, 2))
    return np.vstack([main, nb, uni]), (0.0, 0.0, 100.0)


def fig_circle_ransac():
    print("[11] RANSAC circle")
    p, (cx0, cy0, R) = circle_neighbor_dataset()
    res = F.fit_circle_ransac(p, r_min=80, r_max=120, tol=2.5, iters=3000)
    tb = F.fit_circle_taubin(p)
    tr = F.fit_circle_trimmed(p)
    tk = F.fit_circle_tukey(p)
    info = res.info
    ux, uy, r, valid = info["hyp_all"]
    gate = info["gate"]

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.95]})
    ax = axs[0]
    pts_in(ax, p, MUTED, s=9)
    shown_in = shown_out = 0
    for j in range(len(r)):
        if not valid[j]:
            continue
        if gate[j] and shown_in < 14:
            draw_circle(ax, ux[j], uy[j], r[j], color=INK2, lw=0.8, alpha=0.6)
            shown_in += 1
        elif not gate[j] and shown_out < 14:
            draw_circle(ax, ux[j], uy[j], r[j], color=C["outlier"], lw=0.8, alpha=0.45, ls="--")
            shown_out += 1
    bh = info["best_hyp"]
    draw_circle(ax, bh.cx, bh.cy, bh.r, color=C["ransac"], lw=2.2)
    ax.plot([], [], color=INK2, lw=0.8, label="giả thuyết qua cổng bán kính")
    ax.plot([], [], color=C["outlier"], lw=0.8, ls="--", label="bị cổng [80, 120] loại")
    ax.plot([], [], color=C["ransac"], lw=2.2,
            label=f"tốt nhất: {info['counts'][info['best']]} inlier")
    ax.set_title("(a) Giả thuyết từ bộ 3 điểm ngẫu nhiên")
    equal(ax, (-150, 210), (-150, 150))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=1)

    ax = axs[1]
    th = np.linspace(0, 2 * np.pi, 400)
    ring_o = np.column_stack([res.cx + (res.r + 2.5) * np.cos(th), res.cy + (res.r + 2.5) * np.sin(th)])
    ring_i = np.column_stack([res.cx + (res.r - 2.5) * np.cos(th), res.cy + (res.r - 2.5) * np.sin(th)])
    ax.fill(np.r_[ring_o[:, 0], ring_i[::-1, 0]], np.r_[ring_o[:, 1], ring_i[::-1, 1]],
            color=C["ransac"], alpha=0.14, lw=0)
    pts_in(ax, p[res.inliers], C["ransac"], s=11, label=f"inlier ({res.inliers.sum()})")
    pts_out(ax, p[~res.inliers], s=16, label=f"outlier ({(~res.inliers).sum()})")
    draw_circle(ax, tb.cx, tb.cy, tb.r, color=C["taubin"], ls="--", lw=1.4,
                label=f"Taubin mọi điểm: lệch {math.hypot(tb.cx - cx0, tb.cy - cy0):.1f} pixel")
    draw_circle(ax, tr.cx, tr.cy, tr.r, color=C["trimmed"], ls="-.", lw=1.4,
                label=f"Trimmed: lệch {math.hypot(tr.cx - cx0, tr.cy - cy0):.1f} pixel")
    draw_circle(ax, tk.cx, tk.cy, tk.r, color=C["tukey"], ls=":", lw=1.6,
                label=f"Tukey: lệch {math.hypot(tk.cx - cx0, tk.cy - cy0):.1f} pixel")
    draw_circle(ax, res.cx, res.cy, res.r, color=C["ransac"], lw=1.6,
                label=f"RANSAC: lệch {math.hypot(res.cx - cx0, res.cy - cy0):.2f} pixel")
    ax.set_title("(b) Kết quả so với các phương pháp cục bộ")
    equal(ax, (-150, 210), (-150, 150))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=2)

    ax = axs[2]
    rv = r[valid]
    bins = np.linspace(0, 300, 76)
    ax.hist(np.clip(rv, 0, 300), bins=bins, color=MUTED, edgecolor=SURFACE, linewidth=0.6)
    ax.axvspan(80, 120, color=C["ransac"], alpha=0.14, lw=0, label="cổng bán kính [80, 120]")
    ax.set_xlabel("bán kính giả thuyết (pixel; > 300 gộp vào cột cuối)")
    ax.set_ylabel("số giả thuyết")
    ax.set_title(f"(c) {gate.sum()} / {valid.sum()} giả thuyết qua cổng")
    ax.legend(loc="upper right")
    save(fig, "11_circle_ransac.png")


# ===========================================================================
# Ảnh 12: sai số tâm theo tỉ lệ ngoại lai
# ===========================================================================
def gen_outliers(rng, frac, kind, n=150, R=100.0):
    t = rng.uniform(0, 2 * np.pi, n)
    p = np.column_stack([R * np.cos(t), R * np.sin(t)]) + rng.normal(0, 1.0, (n, 2))
    m = int(round(frac * n))
    if m == 0:
        return p
    if kind == "local":          # khuyết / bụi: biên lệch 5-25% bán kính
        sgn = rng.choice([-1.0, 1.0], m)[:, None]
        p[:m] *= 1 + sgn * rng.uniform(0.05, 0.25, (m, 1))
    else:                        # vật lân cận: một lỗ nhỏ gần đó
        tt = rng.uniform(0, 2 * np.pi, m)
        p[:m] = np.column_stack([170 + 40 * np.cos(tt), 40 * np.sin(tt)]) + rng.normal(0, 1.0, (m, 2))
    return p


def fig_outlier_sweep():
    print("[12] outlier sweep")
    methods = {
        "kasa": ("Kasa", lambda q: F.fit_circle_kasa(q)),
        "taubin": ("Taubin", lambda q: F.fit_circle_taubin(q)),
        "trimmed": ("Trimmed (thang trung vị)", lambda q: F.fit_circle_trimmed(q)),
        "tukey": ("Taubin → Tukey", lambda q: F.fit_circle_tukey(q)),
        "ransac": ("RANSAC [80, 120]", lambda q: F.fit_circle_ransac(q, 80, 120)),
    }
    markers = {"kasa": "o", "taubin": "s", "trimmed": "P", "tukey": "v", "ransac": "D"}
    lss = {"kasa": "-", "taubin": "--", "trimmed": "-.", "tukey": ":", "ransac": "-"}
    fracs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    rng = np.random.default_rng(44)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    table = {}
    for ax, kind, title in ((axs[0], "local", "(a) Ngoại lai cục bộ (khuyết, bụi)"),
                            (axs[1], "cluster", "(b) Ngoại lai dạng cụm (lỗ lân cận)")):
        med = {k: [] for k in methods}
        for fr in fracs:
            e = {k: [] for k in methods}
            for _ in range(150):
                q = gen_outliers(rng, fr, kind)
                for k, (_, fn) in methods.items():
                    f = fn(q)
                    e[k].append(math.hypot(f.cx, f.cy) if f is not None else np.nan)
            for k in methods:
                med[k].append(float(np.nanmedian(e[k])))
        table[kind] = med
        for k, (name, _) in methods.items():
            ax.plot(fracs * 100, med[k], color=C[k], ls=lss[k], marker=markers[k], ms=5, lw=1.6,
                    label=name)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("tỉ lệ ngoại lai (%)")
        ax.set_title(title)
    axs[0].set_ylabel("trung vị sai số tâm (pixel, thang log)")
    axs[1].legend(loc="center right", fontsize=8)
    save(fig, "12_outlier_sweep.png")
    print("  median center error (px)")
    for kind, med in table.items():
        print("  ", kind)
        for k in methods:
            print(f"     {k:8s} " + " ".join(f"{v:7.2f}" for v in med[k]))


# ===========================================================================
# Ảnh 01 + 13: ảnh tổng hợp, caliper xuyên tâm, rồi fit
# ===========================================================================
TRUTH = (181.37, 176.62, 108.4)


def synth_image():
    """Ảnh 360x360: lỗ tối trên nền sáng + vết khuyết + bụi + vật lân cận."""
    h = w = 360
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy, R = TRUTH

    def cover(dist_inside):          # khử răng cưa: phần diện tích pixel nằm trong hình
        return np.clip(dist_inside + 0.5, 0.0, 1.0)

    img = np.full((h, w), 200.0)
    img = img - 150.0 * cover(R - np.hypot(xx - cx, yy - cy))                    # lỗ
    a = np.deg2rad(205)
    nx, ny = cx + R * math.cos(a), cy + R * math.sin(a)
    notch = cover(15.0 - np.hypot(xx - nx, yy - ny))
    img = img * (1 - notch) + 200.0 * notch                                       # vết khuyết sáng
    a = np.deg2rad(315)
    dx, dy = cx + (R + 7) * math.cos(a), cy + (R + 7) * math.sin(a)
    dust = cover(9.0 - np.hypot(xx - dx, yy - dy))
    img = img * (1 - dust) + 50.0 * dust                                          # bụi tối
    rect = cover(np.minimum.reduce([xx - (cx + R + 15), (cx + R + 60) - xx,
                                    yy - (cy - 45), (cy + 45) - yy]))
    img = img * (1 - rect) + 10.0 * rect                                          # vật lân cận
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img += np.random.default_rng(1).normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def radial_edges(img, c0, r0, n_rays=90, r_lo=-30.0, r_hi=45.0, step=0.25):
    """Caliper xuyên tâm: lấy mẫu profile dọc tia, tìm đỉnh |gradient|, nội suy parabol."""
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ts = np.arange(r0 + r_lo, r0 + r_hi, step)
    xs = (c0[0] + np.cos(ang)[:, None] * ts[None, :]).astype(np.float32)
    ys = (c0[1] + np.sin(ang)[:, None] * ts[None, :]).astype(np.float32)
    prof = cv2.remap(img, xs, ys, cv2.INTER_LINEAR)
    k = np.exp(-0.5 * (np.arange(-6, 7) * step / 0.8) ** 2)
    k /= k.sum()
    prof = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, prof)
    g = np.abs(np.gradient(prof, step, axis=1))
    g[:, :8] = g[:, -8:] = 0
    i = g.argmax(axis=1)
    rows = np.arange(n_rays)
    y0, y1, y2 = g[rows, i - 1], g[rows, i], g[rows, i + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    t = ts[i] + np.clip(delta, -1, 1) * step
    pts = np.column_stack([c0[0] + np.cos(ang) * t, c0[1] + np.sin(ang) * t])
    return pts, ang, (xs, ys)


def fig_image_pipeline():
    print("[01/13] synthetic image pipeline")
    img = synth_image()
    c0, r0 = (178.0, 180.0), 104.0                       # ước lượng thô (ví dụ từ Hough)
    pts, ang, (xs, ys) = radial_edges(img, c0, r0)
    cx0, cy0, R = TRUTH
    fits = {
        "kasa": ("Kasa", F.fit_circle_kasa(pts)),
        "taubin": ("Taubin", F.fit_circle_taubin(pts)),
        "trimmed": ("Trimmed", F.fit_circle_trimmed(pts)),
        "tukey": ("Taubin → Tukey", F.fit_circle_tukey(pts)),
        "ransac": ("RANSAC", F.fit_circle_ransac(pts, 90, 125, tol=1.5)),
    }
    rs = fits["ransac"][1]

    # ---- ảnh 01: caliper + điểm biên
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax in axs:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axs[0]
    for j in range(len(ang)):
        ax.plot([xs[j, 0], xs[j, -1]], [ys[j, 0], ys[j, -1]], color="#eda100", lw=0.7, alpha=0.85)
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color="#3987e5", edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_title("(a) 90 caliper xuyên tâm + điểm biên dưới pixel")
    ax = axs[1]
    ax.scatter(pts[rs.inliers, 0], pts[rs.inliers, 1], s=16, color="#1baf7a",
               edgecolors="white", linewidths=0.6, zorder=4, label="inlier của RANSAC")
    ax.scatter(pts[~rs.inliers, 0], pts[~rs.inliers, 1], s=34, marker="x", color=C["outlier"],
               linewidths=1.6, zorder=5, label="outlier")
    draw_circle(ax, rs.cx, rs.cy, rs.r, color="#1baf7a", lw=1.2)
    ax.annotate("vết khuyết", (cx0 + (R - 20) * math.cos(np.deg2rad(205)),
                               cy0 + (R - 20) * math.sin(np.deg2rad(205))),
                xytext=(-10, -40), textcoords="offset points", color="white", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax.annotate("bụi", (cx0 + (R + 16) * math.cos(np.deg2rad(315)), cy0 + (R + 16) * math.sin(np.deg2rad(315))),
                xytext=(20, -10), textcoords="offset points", color="white", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax.annotate("vật lân cận", (cx0 + R + 40, cy0 + 48), xytext=(-50, -45), textcoords="offset points",
                color="white", fontsize=9, arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax.set_title("(b) RANSAC phân loại inlier / outlier")
    ax.legend(loc="lower left", fontsize=8.5, labelcolor="white", facecolor="#333333", frameon=True,
              framealpha=0.7, edgecolor="none")
    save(fig, "01_image_calipers.png")

    # ---- ảnh 13: phân tích kết quả
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axs[0]
    deg = np.rad2deg(ang)
    for k in ("kasa", "ransac"):
        f = fits[k][1]
        ax.plot(deg, f.residuals(pts), color=C[k], marker="o" if k == "kasa" else "D", ms=3.5, lw=1.2,
                label=fits[k][0])
    ax.axhspan(-1.5, 1.5, color=C["ransac"], alpha=0.12, lw=0, label="dải tol của RANSAC")
    for x0, x1, lab in ((195, 215, "khuyết"), (308, 322, "bụi"), (338, 360, "vật\nlân cận"),
                        (0, 22, "vật\nlân cận")):
        ax.axvspan(x0, x1, color=MUTED, alpha=0.10, lw=0)
        ax.text((x0 + x1) / 2, -19, lab, ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(0, 360)
    ax.set_ylim(-20, 24)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("góc tia (độ)")
    ax.set_ylabel("phần dư d = ρ − r  (pixel)")
    ax.set_title("(a) Phần dư theo góc: công cụ debug quan trọng nhất")
    ax.legend(loc="upper center", fontsize=8, ncol=3)

    ax = axs[1]
    keys = list(fits)
    errs = [math.hypot(fits[k][1].cx - cx0, fits[k][1].cy - cy0) for k in keys]
    rerr = [fits[k][1].r - R for k in keys]
    yy = np.arange(len(keys))[::-1]
    ax.barh(yy, errs, height=0.5, color=[C[k] for k in keys])
    for y, e in zip(yy, errs):
        ax.text(e + max(errs) * 0.02, y, f"{e:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([fits[k][0] for k in keys])
    ax.set_xlim(0, max(errs) * 1.2)
    ax.set_xlabel("sai số tâm so với giá trị thật (pixel)")
    ax.set_title("(b) Sai số tâm")
    ax.grid(axis="y", visible=False)

    ax = axs[2]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.grid(False)
    for k in keys:
        f = fits[k][1]
        draw_circle(ax, f.cx, f.cy, f.r, color=C[k] if k != "ransac" else "#1baf7a", lw=1.3,
                    label=fits[k][0])
    draw_circle(ax, cx0, cy0, R, color="white", lw=1, ls=":", label="giá trị thật")
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color="white", zorder=4)
    ax.set_xlim(cx0 + R - 22, cx0 + R + 22)
    ax.set_ylim(cy0 + 30, cy0 - 30)
    ax.set_title("(c) Phóng to vùng vật lân cận (0°)")
    ax.legend(loc="lower right", fontsize=7.5, labelcolor="white", facecolor="#333333", frameon=True,
              framealpha=0.7, edgecolor="none")
    save(fig, "13_image_results.png")
    print("  truth", TRUTH)
    for k in keys:
        f = fits[k][1]
        print(f"  {k:8s} cx={f.cx:8.3f} cy={f.cy:8.3f} r={f.r:8.3f}  center_err={math.hypot(f.cx - cx0, f.cy - cy0):.3f}"
              f"  r_err={f.r - R:+.3f}")


# ===========================================================================
# Bảng số: thời gian chạy + ảnh hưởng của thang đo trong Trimmed
# ===========================================================================
def table_timing():
    print("[T1] timing (ms, median of 30 runs)")
    rng = np.random.default_rng(0)
    for n in (100, 1000):
        t = rng.uniform(0, 2 * np.pi, n)
        p = np.column_stack([100 * np.cos(t), 100 * np.sin(t)]) + rng.normal(0, 1, (n, 2))
        p[: n // 5] = rng.uniform(-150, 150, (n // 5, 2))
        fns = {
            "kasa": lambda: F.fit_circle_kasa(p),
            "taubin": lambda: F.fit_circle_taubin(p),
            "geometric": lambda: F.fit_circle_geometric(p),
            "trimmed": lambda: F.fit_circle_trimmed(p),
            "tukey": lambda: F.fit_circle_tukey(p),
            "ransac(3000)": lambda: F.fit_circle_ransac(p, 80, 120),
        }
        row = []
        for k, fn in fns.items():
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e3)
            row.append(f"{k}={np.median(ts):.2f}")
        print(f"  n={n}: " + "  ".join(row))


def table_trim_scale():
    print("[T2] trimmed scale on clean data (sigma=1, full circle, n=150, 300 trials)")
    rng = np.random.default_rng(9)
    for scale in ("mad", "rms", "std_abs"):
        keep, rerr = [], []
        for _ in range(300):
            t = rng.uniform(0, 2 * np.pi, 150)
            p = np.column_stack([100 * np.cos(t), 100 * np.sin(t)]) + rng.normal(0, 1, (150, 2))
            f = F.fit_circle_trimmed(p, rounds=2, scale=scale)
            keep.append(f.inliers.mean())
            rerr.append(math.hypot(f.cx, f.cy))
        print(f"  {scale:8s} kept={np.mean(keep) * 100:5.1f}%  center_err_rms={math.sqrt(np.mean(np.square(rerr))):.3f}")


def table_ransac_adaptive():
    print("[T3] RANSAC adaptive loop vs vectorised 3000 hypotheses (30% outliers)")
    rng = np.random.default_rng(5)
    t = rng.uniform(0, 2 * np.pi, 200)
    p = np.column_stack([500 + 80 * np.cos(t), 300 + 80 * np.sin(t)]) + rng.normal(0, 0.7, (200, 2))
    p[:60] = rng.uniform(380, 620, (60, 2))

    def fit_minimal(s):
        ux, uy, r, ok = F.circumcircle(s[0], s[1], s[2])
        return (float(ux), float(uy), float(r)) if ok else None

    def residual(m, q):
        return np.abs(np.hypot(q[:, 0] - m[0], q[:, 1] - m[1]) - m[2])

    def refine(q):
        f = F.fit_circle_taubin(q)
        return f.cx, f.cy, f.r

    model, mask, it = F.ransac_generic(p, 3, fit_minimal, residual, refine, tol=2.5)
    vec = F.fit_circle_ransac(p, 60, 100)
    print(f"  adaptive: {it} iterations, {mask.sum()} inliers, circle = {np.round(model, 4)}")
    print(f"  vectorised: {vec.inliers.sum()} inliers, circle = {np.round([vec.cx, vec.cy, vec.r], 4)}")


if __name__ == "__main__":
    fig_image_pipeline()
    fig_line_ols_tls()
    fig_line_tls_svd()
    fig_line_ransac()
    fig_ransac_iterations()
    fig_line_trimmed_tukey()
    fig_weight_functions()
    fig_circle_arc_bias()
    fig_circle_trimmed()
    fig_circumcircle()
    fig_circle_ransac()
    fig_outlier_sweep()
    table_timing()
    table_trim_scale()
    table_ransac_adaptive()
