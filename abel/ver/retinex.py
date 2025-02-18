import cv2
import numpy as np

def single_scale_retinex(image, sigma=50):
    log_image = np.log1p(np.float32(image))
    log_gaussian = np.log1p(np.float32(cv2.GaussianBlur(image, (0, 0), sigma)))
    return cv2.normalize(log_image - log_gaussian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Load the image
image = cv2.imread(r"E:\GIt\project\abel\output_images\set_f38.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Retinex Enhancement
retinex_enhanced = single_scale_retinex(image, sigma=50)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Retinex Enhanced Image", retinex_enhanced)

cv2.waitKey(0)
cv2.destroyAllWindows()
