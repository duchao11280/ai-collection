import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,  help='Image file path')

    args = parser.parse_args()
    img_path = args.image_path

    plt.figure(figsize=(6, 10))
    raw_image = cv2.imread(r"d:\Hachix\project\ECOS_AI-dev\projects\KE_vision\Image__2024-12-03__19-05-55.bmp")

    sc_img = cv2.imread(r"d:\Hachix\project\ECOS_AI-dev\projects\KE_vision\new_2024123.bmp")
    th_img = cv2.imread(r"d:\Hachix\project\ECOS_AI-dev\projects\KE_vision\20241230142932_C3.jpeg")

    histr_blue = cv2.calcHist([raw_image],[0],None,[256],[0,256])
    histr_green = cv2.calcHist([sc_img],[0],None,[256],[0,256])
    histr_red = cv2.calcHist([th_img],[0],None,[256],[0,256])
    # plt.subplot(2, 3, 2)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 4)
    plt.plot(histr_blue)
    plt.title("Blue")
    plt.subplot(2, 3, 5)
    plt.title("Green")
    plt.plot(histr_green)
    plt.subplot(2, 3, 6)
    plt.plot(histr_red)
    plt.title("Red")

    plt.show()