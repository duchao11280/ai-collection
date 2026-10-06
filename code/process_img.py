import cv2
import numpy as np

def clahe_img(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(image_lab)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final_image

def adjust_brightness_contrast_by_value(img, min_value, max_value):
    # normalize
    min_norm = min_value / 255.0
    max_norm = max_value / 255.0

    # Calculate brightness and contrast
    contrast = 1 / (max_norm - min_norm)
    brightness = -min_norm * contrast * 255
    print(contrast, brightness)
    img_adjusted = cv2.addWeighted(img, contrast, img, 0, brightness)
    return img_adjusted

if __name__ == "__main__":
    img = cv2.imread(r"D:\Hachix\project\ECOS_AI-dev\projects\KE_vision\temp\C1\20241206125929_C1.jpeg")
    # image = adjust_brightness_contrast_by_value(img, 85, 150)
    # image = adjust_brightness_contrast_by_value(img, 47, 88)
    # image = adjust_brightness_contrast_by_value(img, 88, 125)
    # img = clahe_img(img)
    # image = adjust_brightness_contrast_by_value(img, 34, 88)
    image = adjust_brightness_contrast_by_value(img, 164, 255)
    cv2.imwrite("check.png", image)