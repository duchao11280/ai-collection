"""
Tái hiện Fiji: Image > Color Threshold (HSB)
Thông số từ screenshot:
  Hue        : 87  – 183  (thang Fiji 0-255)
  Saturation : 0   – 255
  Brightness : 52  – 238
  Dark background: True
"""

import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# 1. Hàm chuyển đổi thang Fiji (0-255) → OpenCV
#    Fiji Hue: 0-255  |  OpenCV Hue: 0-179
# ──────────────────────────────────────────────
def fiji_hue_to_cv(h_fiji: int) -> int:
    return round(h_fiji * 179 / 255)


# ──────────────────────────────────────────────
# 2. Thông số threshold (chỉnh theo nhu cầu)
# ──────────────────────────────────────────────
HUE_MIN        = fiji_hue_to_cv(87)   # ≈ 61
HUE_MAX        = fiji_hue_to_cv(183)  # ≈ 128
SAT_MIN        = 0
SAT_MAX        = 255
BRIGHT_MIN     = 52
BRIGHT_MAX     = 238
DARK_BACKGROUND = False   # Fiji "Dark background" → đảo mask


# ──────────────────────────────────────────────
# 3. Đọc ảnh và crop (tuỳ chỉnh vùng crop)
# ──────────────────────────────────────────────
def load_and_crop(image_path: str,
                  crop_rect: tuple | None = None) -> np.ndarray:
    """
    crop_rect: (x, y, w, h) tính từ góc trên-trái, hoặc None để dùng toàn ảnh.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    if crop_rect is not None:
        x, y, w, h = crop_rect
        img = img[y:y+h, x:x+w]
    return img


# ──────────────────────────────────────────────
# 4. Áp dụng Color Threshold theo HSV
# ──────────────────────────────────────────────
def color_threshold_hsv(bgr_img: np.ndarray,
                        h_min, h_max,
                        s_min, s_max,
                        v_min, v_max,
                        dark_background: bool = False) -> np.ndarray:
    """
    Trả về binary mask (255 = vùng được chọn).
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    if dark_background:
        # Fiji "Dark background": đảo ngược mask
        mask = cv2.bitwise_not(mask)

    return mask


# ──────────────────────────────────────────────
# 5. Vẽ contour đỏ lên ảnh (giống Fiji)
# ──────────────────────────────────────────────
def draw_contours(bgr_img: np.ndarray, mask: np.ndarray,
                  color=(0, 0, 255), thickness=2) -> np.ndarray:
    result = bgr_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


# ──────────────────────────────────────────────
# 6. Pipeline chính
# ──────────────────────────────────────────────
def process(image_path: str,
            crop_rect: tuple | None = None,
            output_dir: str = ".") -> None:
    # Đọc & crop
    img = load_and_crop(image_path, crop_rect)

    # Tạo mask
    mask = color_threshold_hsv(
        img,
        h_min=HUE_MIN,    h_max=HUE_MAX,
        s_min=SAT_MIN,    s_max=SAT_MAX,
        v_min=BRIGHT_MIN, v_max=BRIGHT_MAX,
        dark_background=DARK_BACKGROUND,
    )

    # Vẽ contour đỏ
    result = draw_contours(img, mask)

    # Lưu kết quả
    stem = Path(image_path).stem
    out_mask    = str(Path(output_dir) / f"{stem}_mask.png")
    out_contour = str(Path(output_dir) / f"{stem}_contour.png")

    cv2.imwrite(out_mask, mask)
    cv2.imwrite(out_contour, result)

    print(f"[OK] Mask     → {out_mask}")
    print(f"[OK] Contour  → {out_contour}")

    # Hiển thị (tuỳ chọn)
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Contour (red)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# 7. Chạy thử
# ──────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = r"E:\Hachix\data\nitto\Nitto_20240416\Film-vang-sau-khi-dan-NG\Image_20260416013243318.bmp"   # ← đổi thành đường dẫn ảnh của bạn

    # crop_rect = (x, y, w, h)  ← đổi vùng crop cho phù hợp, hoặc để None
    CROP_RECT  = (1142, 550, 500, 500)  # Ví dụ: crop vùng 400x400 từ (1142, 550)

    process(IMAGE_PATH, crop_rect=CROP_RECT, output_dir=".")