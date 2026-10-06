#!/usr/bin/env python3
"""Fitting đường thẳng và đường tròn từ tập điểm biên (chỉ cần numpy).

Các thuật toán:

* Đường thẳng: bình phương tối thiểu thông thường (y = a*x + b),
  Total Least Squares (hồi quy trực giao), RANSAC, Trimmed, Tukey tái trọng số.
* Đường tròn: Kasa (bình phương tối thiểu đại số), Taubin, fit hình học
  (Levenberg-Marquardt, dùng làm tham chiếu), Trimmed, RANSAC, Tukey tái trọng số.

Mọi hàm fit trả về ``LineFit`` hoặc ``CircleFit``; trường ``info`` chứa dữ liệu
trung gian (các vòng lặp, các giả thuyết RANSAC...) để vẽ ảnh debug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Kiểu dữ liệu kết quả
# ---------------------------------------------------------------------------
@dataclass
class LineFit:
    p0: np.ndarray                  # một điểm trên đường thẳng
    direction: np.ndarray           # vector chỉ phương đơn vị
    inliers: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def residuals(self, points) -> np.ndarray:
        """Khoảng cách có dấu (trực giao) từ từng điểm tới đường thẳng."""
        return (as_xy(points) - self.p0) @ self.normal


@dataclass
class CircleFit:
    cx: float
    cy: float
    r: float
    inliers: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)

    def residuals(self, points) -> np.ndarray:
        """Khoảng cách hình học có dấu: dương = điểm nằm ngoài đường tròn."""
        p = as_xy(points)
        return np.hypot(p[:, 0] - self.cx, p[:, 1] - self.cy) - self.r


def as_xy(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _weights(n: int, weights) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)


# ---------------------------------------------------------------------------
# Thang đo nhiễu và hàm trọng số
# ---------------------------------------------------------------------------
def mad_scale(res: np.ndarray) -> float:
    """Độ lệch chuẩn ước lượng bền vững: 1.4826 * median(|r - median(r)|).

    Hệ số 1.4826 làm cho kết quả bằng đúng sigma khi nhiễu là Gauss, và giá trị
    này không bị kéo bởi tới 50% điểm ngoại lai (khác với np.std).
    """
    res = np.asarray(res, dtype=np.float64)
    return 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-12


def tukey_weights(u: np.ndarray) -> np.ndarray:
    """Tukey biweight: w = (1 - u^2)^2 khi |u| < 1, ngược lại bằng 0."""
    return np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)


def huber_weights(u: np.ndarray, k: float = 1.345) -> np.ndarray:
    """Huber: w = 1 khi |u| <= k, ngược lại k/|u| (không bao giờ bằng 0)."""
    a = np.abs(u)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def ransac_iterations(p_success: float, inlier_ratio: float, sample_size: int) -> int:
    """Số lần lấy mẫu để với xác suất p_success có ít nhất một mẫu sạch."""
    w_s = inlier_ratio ** sample_size
    if w_s >= 1.0:
        return 1
    if w_s <= 0.0:
        return 10 ** 9
    return int(math.ceil(math.log(1.0 - p_success) / math.log(1.0 - w_s)))


# ===========================================================================
# ĐƯỜNG THẲNG
# ===========================================================================
def fit_line_ols(points) -> tuple:
    """y = a*x + b, sai số đo theo phương thẳng đứng. Trả về (a, b).

    Chỉ dùng để so sánh: hỏng với đường gần thẳng đứng.
    """
    p = as_xy(points)
    A = np.column_stack([p[:, 0], np.ones(len(p))])
    (a, b), *_ = np.linalg.lstsq(A, p[:, 1], rcond=None)
    return float(a), float(b)


def fit_line_tls(points, weights=None) -> LineFit:
    """Total Least Squares (hồi quy trực giao) bằng phân tích giá trị suy biến.

    Cực tiểu sum w_i * ((p_i - p0) . n)^2 với |n| = 1:
      * p0 tối ưu = trọng tâm có trọng số của các điểm,
      * n tối ưu  = vector suy biến phải ứng với giá trị suy biến nhỏ nhất,
        tức hướng mà dữ liệu "mỏng" nhất; chỉ phương là vector còn lại.
    """
    p = as_xy(points)
    if len(p) < 2:
        raise ValueError("fit_line_tls cần ít nhất 2 điểm")
    w = _weights(len(p), weights)
    p0 = (w[:, None] * p).sum(axis=0) / w.sum()          # trọng tâm dùng w, không phải sqrt(w)
    q = (p - p0) * np.sqrt(w)[:, None]                   # mỗi hàng nhân sqrt(w) -> hiệp phương sai có trọng số w
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    return LineFit(p0=p0, direction=vt[0], info={"singular_values": s})


def fit_line_ransac(points, tol: float = 2.0, iters: int = 500,
                    rng: Optional[np.random.Generator] = None) -> Optional[LineFit]:
    """RANSAC cho đường thẳng, vector hoá toàn bộ các giả thuyết.

    1. Lấy ngẫu nhiên ``iters`` cặp điểm -> mỗi cặp là một giả thuyết.
    2. Tính khoảng cách của MỌI điểm tới MỌI giả thuyết (ma trận iters x n).
    3. Giả thuyết có nhiều điểm trong dải ``tol`` nhất thắng.
    4. Fit lại Total Least Squares trên các điểm trong (inlier) của nó.
    """
    p = as_xy(points)
    n = len(p)
    if n < 2:
        return None
    rng = np.random.default_rng(0) if rng is None else rng   # cố định seed -> kết quả lặp lại được
    idx = rng.integers(0, n, size=(int(iters), 2))
    a, b = p[idx[:, 0]], p[idx[:, 1]]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-9                                        # loại cặp trùng điểm
    if not ok.any():
        return None
    a, d, length, idx = a[ok], d[ok], length[ok], idx[ok]
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / length[:, None]
    # |(p - a) . n|: lấy trị tuyệt đối SAU tích vô hướng.
    dist = np.abs(((p[None, :, :] - a[:, None, :]) * normal[:, None, :]).sum(-1))
    counts = (dist < tol).sum(axis=1)
    best = int(counts.argmax())
    mask = dist[best] < tol
    if mask.sum() < 2:
        return None
    fit = fit_line_tls(p[mask])
    # Thu lại inlier theo đường đã tinh chỉnh (ổn định hơn đường qua 2 điểm).
    mask = np.abs(fit.residuals(p)) < tol
    fit = fit_line_tls(p[mask])
    fit.inliers = mask
    fit.info = {"pairs": idx, "counts": counts, "best": best,
                "best_p0": a[best], "best_dir": d[best] / length[best]}
    return fit


def fit_line_trimmed(points, rounds: int = 3, k: float = 2.5,
                     scale: str = "mad", min_points: int = 5) -> LineFit:
    """Total Least Squares + cắt tỉa lặp: bỏ điểm có |d| > k * s rồi fit lại.

    ``scale``: cách ước lượng s từ phần dư
      * ``"mad"``     : 1.4826 * median(|d|) trên mọi điểm (khuyến nghị),
      * ``"rms"``     : căn quân phương của |d| trên các điểm đang giữ,
      * ``"std_abs"`` : np.std(|d|) trên các điểm đang giữ (hay gặp, nhưng
                        chỉ bằng ~0.6 sigma -> cắt cả điểm tốt).
    """
    p = as_xy(points)
    keep = np.ones(len(p), dtype=bool)
    fit = fit_line_tls(p)
    history = [(fit.p0.copy(), fit.direction.copy(), keep.copy())]
    for _ in range(int(rounds)):
        d = np.abs(fit.residuals(p))
        s = _trim_scale(d, keep, scale)
        if s < 1e-9:
            break
        new_keep = d <= k * s
        if new_keep.sum() < min_points or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        fit = fit_line_tls(p[keep])
        history.append((fit.p0.copy(), fit.direction.copy(), keep.copy()))
    fit.inliers = keep
    fit.info = {"history": history}
    return fit


def fit_line_reject_n(points, reject_n: int) -> LineFit:
    """Biến thể đơn giản của trimmed: bỏ đúng ``reject_n`` điểm tệ nhất, fit lại."""
    p = as_xy(points)
    fit = fit_line_tls(p)
    if reject_n <= 0 or len(p) - reject_n < 2:
        fit.inliers = np.ones(len(p), dtype=bool)
        return fit
    order = np.argsort(np.abs(fit.residuals(p)))
    keep = np.zeros(len(p), dtype=bool)
    keep[order[: len(p) - int(reject_n)]] = True
    fit = fit_line_tls(p[keep])
    fit.inliers = keep
    return fit


def fit_line_tukey(points, init: Optional[LineFit] = None, c: float = 4.685,
                   iters: int = 10) -> LineFit:
    """Tinh chỉnh bền vững bằng tái trọng số lặp với hàm Tukey biweight."""
    p = as_xy(points)
    fit = init if init is not None else fit_line_tls(p)
    w = np.ones(len(p))
    for _ in range(int(iters)):
        res = fit.residuals(p)
        s = mad_scale(res)
        w = tukey_weights(res / (c * s))
        if (w > 0).sum() < 2:
            break
        fit = fit_line_tls(p, weights=w)
    fit.inliers = w > 0
    fit.info = {"weights": w}
    return fit


# ===========================================================================
# ĐƯỜNG TRÒN
# ===========================================================================
def fit_circle_kasa(points, weights=None) -> CircleFit:
    """Bình phương tối thiểu đại số (Kasa, 1976).

    Viết lại (x-a)^2 + (y-b)^2 = r^2 thành  2a*x + 2b*y + c = x^2 + y^2
    với c = r^2 - a^2 - b^2  ->  hệ TUYẾN TÍNH theo (a, b, c).
    Dữ liệu được trừ trọng tâm trước để ma trận không bị "ill-conditioned".
    """
    p = as_xy(points)
    if len(p) < 3:
        raise ValueError("fit_circle_kasa cần ít nhất 3 điểm")
    w = _weights(len(p), weights)
    m = (w[:, None] * p).sum(axis=0) / w.sum()
    x, y = p[:, 0] - m[0], p[:, 1] - m[1]
    sw = np.sqrt(w)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))]) * sw[:, None]
    b = (x * x + y * y) * sw
    (a0, b0, c0), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c0 + a0 * a0 + b0 * b0, 0.0))
    return CircleFit(float(a0 + m[0]), float(b0 + m[1]), float(r))


def fit_circle_taubin(points, weights=None) -> CircleFit:
    """Taubin (1991), dạng giải bằng phân tích giá trị suy biến (Chernov).

    Phương trình tổng quát:  A*z + B*u + C*v + D = 0,  z = u^2 + v^2
    (u, v là toạ độ đã trừ trọng tâm). Ràng buộc Taubin chuẩn hoá theo
    trung bình bình phương gradient:  4*A^2*mean(z) + B^2 + C^2 = 1.

    Khử D (= -A*mean(z)) và đổi biến A' = 2*sqrt(mean(z))*A thì ràng buộc
    thành |(A', B, C)| = 1 -> nghiệm là vector suy biến phải nhỏ nhất của
    ma trận [ (z - mean(z)) / (2 sqrt(mean(z))),  u,  v ].
    """
    p = as_xy(points)
    if len(p) < 3:
        raise ValueError("fit_circle_taubin cần ít nhất 3 điểm")
    w = _weights(len(p), weights)
    m = (w[:, None] * p).sum(axis=0) / w.sum()
    u, v = p[:, 0] - m[0], p[:, 1] - m[1]
    z = u * u + v * v
    zmean = float((w * z).sum() / w.sum())
    if zmean <= 0:
        return fit_circle_kasa(p, weights)
    z0 = (z - zmean) / (2.0 * math.sqrt(zmean))
    M = np.column_stack([z0, u, v]) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    a_prime, B, C = vt[-1]
    A = a_prime / (2.0 * math.sqrt(zmean))
    if abs(A) < 1e-14:                       # các điểm gần thẳng hàng: bán kính -> vô cùng
        return fit_circle_kasa(p, weights)
    cu, cv = -B / (2.0 * A), -C / (2.0 * A)
    r = math.sqrt(max(cu * cu + cv * cv + zmean, 0.0))
    return CircleFit(float(cu + m[0]), float(cv + m[1]), float(r))


def fit_circle_geometric(points, init: Optional[CircleFit] = None,
                         iters: int = 50, weights=None) -> CircleFit:
    """Fit hình học: cực tiểu sum (||p_i - c|| - r)^2 bằng Levenberg-Marquardt.

    Không có nghiệm đóng -> cần điểm khởi tạo (mặc định lấy Taubin).
    Dùng làm "chuẩn vàng" để đo độ lệch của các phương pháp đại số.
    """
    p = as_xy(points)
    f = init if init is not None else fit_circle_taubin(p, weights)
    w = _weights(len(p), weights)
    theta = np.array([f.cx, f.cy, f.r])
    lam = 1e-3

    def cost(t):
        res = np.hypot(p[:, 0] - t[0], p[:, 1] - t[1]) - t[2]
        return float((w * res * res).sum())

    cur = cost(theta)
    for _ in range(int(iters)):
        dx, dy = p[:, 0] - theta[0], p[:, 1] - theta[1]
        rho = np.maximum(np.hypot(dx, dy), 1e-12)
        res = rho - theta[2]
        J = np.column_stack([-dx / rho, -dy / rho, -np.ones(len(p))])
        JtW = J.T * w
        H, g = JtW @ J, JtW @ res
        step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
        cand = theta + step
        new = cost(cand)
        if new < cur:
            theta, cur, lam = cand, new, lam * 0.3
            if np.abs(step).max() < 1e-10:
                break
        else:
            lam *= 10.0
    return CircleFit(float(theta[0]), float(theta[1]), float(abs(theta[2])))


_CIRCLE_FITS: dict = {
    "kasa": fit_circle_kasa,
    "taubin": fit_circle_taubin,
}


def fit_circle_trimmed(points, rounds: int = 3, k: float = 2.5, scale: str = "mad",
                       inner: str = "taubin", min_points: int = 6) -> Optional[CircleFit]:
    """Fit + cắt tỉa lặp: mỗi vòng bỏ các điểm có |d| > k * s rồi fit lại.

    ``inner``: phương pháp fit bên trong ("taubin" hoặc "kasa").
    ``scale``: xem :func:`fit_line_trimmed`.
    """
    p = as_xy(points)
    if len(p) < min_points:
        return None
    fit_fn = _CIRCLE_FITS[inner]
    keep = np.ones(len(p), dtype=bool)
    fit = fit_fn(p)
    history = [(fit.cx, fit.cy, fit.r, keep.copy())]
    for _ in range(int(rounds)):
        d = np.abs(fit.residuals(p))
        s = _trim_scale(d, keep, scale)
        if s < 1e-9:
            break
        new_keep = d <= k * s
        if new_keep.sum() < min_points or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        fit = fit_fn(p[keep])
        history.append((fit.cx, fit.cy, fit.r, keep.copy()))
    fit.inliers = keep
    fit.info = {"history": history}
    return fit


def circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Đường tròn ngoại tiếp tam giác (a, b, c), vector hoá theo hàng.

    Tâm là giao của hai đường trung trực. Giải hệ 2x2 bằng quy tắc Cramer:
      d  = 2 * (ax(by - cy) + bx(cy - ay) + cx(ay - by))
      ux = (|a|^2 (by - cy) + |b|^2 (cy - ay) + |c|^2 (ay - by)) / d
      uy = (|a|^2 (cx - bx) + |b|^2 (ax - cx) + |c|^2 (bx - ax)) / d
    d = 0 khi ba điểm thẳng hàng (hoặc trùng nhau) -> trả về valid = False.
    """
    ax, ay = a[..., 0], a[..., 1]
    bx, by = b[..., 0], b[..., 1]
    cx, cy = c[..., 0], c[..., 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    valid = np.abs(d) > 1e-9
    d = np.where(valid, d, 1.0)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return ux, uy, r, valid


def fit_circle_ransac(points, r_min: Optional[float] = None, r_max: Optional[float] = None,
                      tol: float = 2.5, iters: int = 3000, chunk: int = 512,
                      inner: str = "taubin",
                      rng: Optional[np.random.Generator] = None) -> Optional[CircleFit]:
    """RANSAC cho đường tròn, vector hoá, có cổng bán kính [r_min, r_max].

    1. Lấy ``iters`` bộ 3 điểm, tính đường tròn ngoại tiếp của tất cả cùng lúc.
    2. Bỏ bộ thẳng hàng và bộ có bán kính ngoài [r_min, r_max] (tri thức có sẵn
       về kích thước lỗ -> loại rất nhiều giả thuyết rác gần như miễn phí).
    3. Đếm inlier theo từng khối ``chunk`` giả thuyết để giới hạn bộ nhớ
       (ma trận chunk x n thay vì iters x n).
    4. Fit lại (Taubin) trên inlier của giả thuyết tốt nhất, thu inlier lần nữa.
    """
    p = as_xy(points)
    n = len(p)
    if n < 3:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, n, size=(int(iters), 3))
    ux, uy, r, valid = circumcircle(p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]])
    gate = valid.copy()
    if r_min is not None:
        gate &= r >= r_min
    if r_max is not None:
        gate &= r <= r_max
    if not gate.any():
        return None
    hx, hy, hr = ux[gate], uy[gate], r[gate]
    px, py = p[:, 0][None, :], p[:, 1][None, :]
    counts = np.empty(len(hx), dtype=np.int64)
    for s in range(0, len(hx), int(chunk)):
        e = s + int(chunk)
        dist = np.abs(np.hypot(px - hx[s:e, None], py - hy[s:e, None]) - hr[s:e, None])
        counts[s:e] = (dist < tol).sum(axis=1)
    best = int(counts.argmax())
    best_hyp = CircleFit(float(hx[best]), float(hy[best]), float(hr[best]))
    mask = np.abs(best_hyp.residuals(p)) < tol
    if mask.sum() < 3:
        return None
    fit_fn = _CIRCLE_FITS[inner]
    fit = fit_fn(p[mask])
    mask = np.abs(fit.residuals(p)) < tol
    if mask.sum() >= 3:
        fit = fit_fn(p[mask])
    fit.inliers = mask
    fit.info = {"hyp_all": (ux, uy, r, valid), "gate": gate, "triples": idx,
                "counts": counts, "best": best, "best_hyp": best_hyp}
    return fit


def fit_circle_tukey(points, init: Optional[CircleFit] = None, inner: str = "taubin",
                     c: float = 4.685, iters: int = 10) -> CircleFit:
    """Tinh chỉnh bền vững bằng tái trọng số lặp (Tukey biweight).

    ``inner`` là phương pháp fit CÓ TRỌNG SỐ dùng ở mỗi vòng. Dùng "kasa" ở
    đây sẽ đưa lại đúng độ lệch bán kính của Kasa trên cung ngắn, kể cả khi
    điểm khởi tạo là Taubin.
    """
    p = as_xy(points)
    fit_fn = _CIRCLE_FITS[inner]
    fit = init if init is not None else fit_circle_taubin(p)
    w = np.ones(len(p))
    for _ in range(int(iters)):
        res = fit.residuals(p)
        s = mad_scale(res)
        w = tukey_weights(res / (c * s))
        if (w > 0).sum() < 6:
            break
        fit = fit_fn(p, weights=w)
    fit.inliers = w > 0
    fit.info = {"weights": w}
    return fit


# ---------------------------------------------------------------------------
# RANSAC tổng quát (dạng vòng lặp dễ đọc, số vòng thích nghi)
# ---------------------------------------------------------------------------
def ransac_generic(points, sample_size: int, fit_minimal: Callable, residual_fn: Callable,
                   refine: Callable, tol: float, p_success: float = 0.99,
                   max_iters: int = 10000, rng: Optional[np.random.Generator] = None):
    """Khung RANSAC dùng chung cho mọi mô hình.

    fit_minimal(sample) -> model hoặc None   (mô hình từ mẫu tối thiểu)
    residual_fn(model, points) -> |d|         (khoảng cách của mọi điểm)
    refine(points_inlier) -> model            (fit lại trên toàn bộ inlier)
    """
    p = as_xy(points)
    n = len(p)
    rng = np.random.default_rng(0) if rng is None else rng
    best_model, best_mask, best_count = None, None, 0
    needed, it = max_iters, 0
    while it < min(needed, max_iters):
        it += 1
        sample = p[rng.choice(n, size=sample_size, replace=False)]
        model = fit_minimal(sample)
        if model is None:
            continue
        mask = residual_fn(model, p) < tol
        count = int(mask.sum())
        if count > best_count:
            best_model, best_mask, best_count = model, mask, count
            # Cập nhật số vòng cần thiết theo tỉ lệ inlier tốt nhất đã thấy.
            needed = ransac_iterations(p_success, count / n, sample_size)
    if best_model is None:
        return None, None, it
    model = refine(p[best_mask])
    mask = residual_fn(model, p) < tol
    return refine(p[mask]), mask, it


# ---------------------------------------------------------------------------
def _trim_scale(d: np.ndarray, keep: np.ndarray, scale: str) -> float:
    if scale == "mad":
        return 1.4826 * float(np.median(d))            # d >= 0 và có median ~ 0.674 sigma
    dk = d[keep] if keep.sum() > 5 else d
    if scale == "rms":
        return float(np.sqrt(np.mean(dk * dk)))
    if scale == "std_abs":
        return float(np.std(dk))
    raise ValueError(f"scale không hợp lệ: {scale}")


__all__ = [
    "LineFit", "CircleFit", "as_xy", "mad_scale", "tukey_weights", "huber_weights",
    "ransac_iterations",
    "fit_line_ols", "fit_line_tls", "fit_line_ransac", "fit_line_trimmed",
    "fit_line_reject_n", "fit_line_tukey",
    "fit_circle_kasa", "fit_circle_taubin", "fit_circle_geometric",
    "fit_circle_trimmed", "circumcircle", "fit_circle_ransac", "fit_circle_tukey",
    "ransac_generic",
]
