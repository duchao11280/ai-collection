import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(image1_path, image2_path):
    # Đọc ảnh
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        raise ValueError("One or both image paths are invalid.")

    # Chuyển ảnh sang không gian màu HSV
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Tính histogram cho từng kênh
    histr_gray_1 = cv2.calcHist([img1],[0],None,[256],[0,256])

    histr_gray_2 = cv2.calcHist([img2],[0],None,[256],[0,256])


    # Vẽ đồ thị histogram
    plt.figure(figsize=(12, 6))

    # Hue channel
    plt.subplot(2, 2, 1)
    plt.plot(histr_gray_1)
    plt.legend()

    # Saturation channel
    plt.subplot(2, 2, 2)
    plt.plot(histr_gray_2)
    plt.legend()


    plt.tight_layout()
    plt.show()

# Đường dẫn ảnh
image1_path = r"D:\Hachix\note_task\alphavina\korea\04122024_C1_C2\C2\20241204141014_C2.jpeg"
image2_path = r"D:\Hachix\project\ECOS_AI-dev\projects\KE_vision\temp\temp_moi\C2\20241206141713_C2.jpeg"

# Gọi hàm vẽ đồ thị histogram
plot_histograms(image1_path, image2_path)