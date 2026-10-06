import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
# https://github.com/maunesh/opencv-gui-helper-tool

def countour_integration():
    # Load the image
    img = cv2.imread('paper_on_black_background.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize area
    area = 0

    # Iterate through contours and calculate area
    for contour in contours:
        area += cv2.contourArea(contour)

    print("Area:", area)


def thresholding_and_shape_analysis():
    # Load the image
    file_path = r'D:\Hachix\project\Healthcare_BLOOD_SERUM\test\IMG_2937.JPG'
    filename = os.path.basename(file_path)
    img = cv2.imread(file_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Remove noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize area
    area = 0

    # Iterate through contours and calculate area
    for contour in contours:
        area += cv2.contourArea(contour)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    if large_contours:
        cv2.drawContours(img, large_contours, -1, (0, 0, 255), 2)
        cv2.imwrite("check/bbox_"+filename,img)
    print("Area:", area)


# def active_contours_snakes():
#     # Load the image
#     img = cv2.imread('paper_on_black_background.jpg')

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Initialize snake parameters
#     snake = np.zeros((1, 1, 2), dtype=np.float32)
#     snake[0, 0, 0] = 0
#     snake[0, 0, 1] = 0

#     # Define energy function
#     def energy(img, snake):
#         energy = 0
#         for i in range(snake.shape[0]):
#             x, y = snake[i, 0, 0], snake[i, 0, 1]
#             energy += (img[int(y), int(x)] - 0) ** 2
#         return energy

#     # Evolve the snake
#     for i in range(100):
#         # Calculate gradient of energy
#         gradient = np.zeros((snake.shape[0], 2), dtype=np.float32)
#         for j in range(snake.shape[0]):
#             x, y = snake[j, 0, 0], snake[j, 0, 1]
#             gradient[j, 0] = (img[int(y), int(x+1)] - img[int(y), int(x-1)]) / 2
#             gradient[j, 1] = (img[int(y+1), int(x)] - img[int(y-1), int(x)]) / 2

#         # Update snake position
#         snake += 0.1 * gradient

#     # Calculate area
#     area = 0
#     for i in range(snake.shape[0]):
#         x, y = snake[i, 0, 0], snake[i, 0, 1]
#         area += (x - x) * (y - y)

#     print("Area:", area)


def snake_scikit():
    # Load image and convert to float
    image = cv2.imread(r'D:\Hachix\project\Healthcare_BLOOD_SERUM\test\IMG_2937.JPG')  # Sử dụng ảnh mẫu 'astronaut' từ scikit-image
    gray_image = image[:,:,0]  # Sử dụng kênh màu đỏ để chuyển thành ảnh xám

    # Smooth the image
    smoothed_image = gaussian(gray_image, 3)

    # Tạo đường viền ban đầu (initial contour)
    # s = np.linspace(0, 2*np.pi, 400)
    rows, cols = gray_image.shape
    x = np.linspace(0, cols-1, 100)
    y = np.linspace(0, rows-1, 100)
    x, y = np.meshgrid(x, y)
    init = np.array([x.ravel(), y.ravel()]).T

    # Áp dụng thuật toán Snake
    snake = active_contour(smoothed_image, init, alpha=0.015, beta=10, gamma=0.001)

    # Plot kết quả
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray_image, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3, label='Initial contour')
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3, label='Snake contour')
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend()
    plt.show()

def laplacian():
    
    # Đọc ảnh
    img = cv2.imread(r'D:\Hachix\project\Healthcare_BLOOD_SERUM\test\IMG_2937.JPG')

    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gaussian để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Áp dụng toán tử Laplacian
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)

    # Chuyển đổi kết quả về dạng ảnh uint8
    laplacian = np.uint8(np.absolute(laplacian))

    # Hiển thị kết quả
    cv2.imwrite('Laplacian.png', laplacian)


# laplacian()
import cv2
import numpy as np

# Tạo một ảnh đơn giản với một hình chữ nhật trắng trên nền đen
image = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)

# Tìm contour của hình chữ nhật
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(image[150][150])
# Vẽ contour lên ảnh để xem các điểm contour
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# Hiển thị kết quả
cv2.imshow('Contour Image', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()